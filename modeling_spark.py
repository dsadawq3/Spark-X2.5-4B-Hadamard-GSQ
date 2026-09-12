# coding=utf-8
# Copyright 2026 The XHToken team and F-Labs Quantization Architecture. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import scipy.linalg
import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import TransformersKwargs, can_return_tuple, logging

try:
    from .configuration_spark import Spark2_5Config
except ImportError:
    from configuration_spark import Spark2_5Config

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "Spark2_5Config"


# ==============================================================================
# F-LABS HADAMARD-SPIN GROUP-WISE INT4 + SRC LOW-RANK SVD LINEAR LAYER
# ==============================================================================
class HadamardGSQLinear(nn.Module):
    """
    Production-grade compressed Linear module implementing:
      - Orthonormal Walsh-Hadamard spin rotation for activation and weight outlier elimination
      - Group-wise INT4 quantization with symmetric group scaling (group_size=64)
      - SVD Residual Compensation (SRC) Low-Rank SVD (r in {16, 32})
      - Invertible algebraic reconstruction: W_eff = (dequantize(Q, s) + A @ B) @ H
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 64,
        rank: int = 16,
        bias: bool = False,
        hadamard: bool = True,
        hadamard_dim: int = 256,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.rank = rank
        self.hadamard = hadamard
        self.hadamard_dim = hadamard_dim

        # Packed INT4 buffer: 2 values per uint8 byte
        self.register_buffer(
            "qweight",
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
            persistent=True,
        )
        # Group scales in BF16
        self.register_buffer(
            "scales",
            torch.zeros(out_features, in_features // group_size, dtype=torch.bfloat16),
            persistent=True,
        )
        # SVD Low-rank residual factors in BF16
        self.register_buffer(
            "res_u",
            torch.zeros(out_features, rank, dtype=torch.bfloat16),
            persistent=True,
        )
        self.register_buffer(
            "res_v",
            torch.zeros(rank, in_features, dtype=torch.bfloat16),
            persistent=True,
        )

        # DV-SSQ: Multi-Precision Subspace Salience (INT8 Tier)
        num_salient = in_features // 8 if in_features >= 8 else 0
        self.num_salient = num_salient
        if num_salient > 0:
            self.register_buffer(
                "salient_idx",
                torch.zeros(num_salient, dtype=torch.int32),
                persistent=True,
            )
            self.register_buffer(
                "salient_weight",
                torch.zeros(out_features, num_salient, dtype=torch.int8),
                persistent=True,
            )
            self.register_buffer(
                "salient_scale",
                torch.zeros(1, num_salient, dtype=torch.bfloat16),
                persistent=True,
            )
        else:
            self.salient_idx = None
            self.salient_weight = None
            self.salient_scale = None

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16))
        else:
            self.register_parameter("bias", None)

        self._cached_weight = None
        self._hadamard_matrix = None

    class _WeightProxy:
        """Lightweight descriptor providing .dtype and .shape without full dequantization."""
        def __init__(self, parent):
            self._parent = parent

        @property
        def dtype(self):
            return self._parent.scales.dtype

        @property
        def shape(self):
            return torch.Size([self._parent.out_features, self._parent.in_features])

        @property
        def device(self):
            return self._parent.qweight.device

    @property
    def weight(self):
        if self._cached_weight is not None:
            return self._cached_weight
        return self._WeightProxy(self)

    def get_hadamard_matrix(self, device):
        if self._hadamard_matrix is None or self._hadamard_matrix.device != device:
            h_np = scipy.linalg.hadamard(self.hadamard_dim)
            h_torch = torch.from_numpy(h_np).to(device=device, dtype=torch.float32) / math.sqrt(self.hadamard_dim)
            self._hadamard_matrix = h_torch
        return self._hadamard_matrix

    def dequantize(self, unrotate: bool = True) -> torch.Tensor:
        """
        Dequantizes INT4 weights, overrides salient channels with INT8 precision,
        adds SVD low-rank residual, and applies inverse Hadamard rotation.
        """
        # 1. Unpack INT4 nibbles
        v0 = (self.qweight & 0x0F).to(torch.int8)
        v1 = ((self.qweight >> 4) & 0x0F).to(torch.int8)
        q0 = torch.where(v0 >= 8, v0 - 16, v0)
        q1 = torch.where(v1 >= 8, v1 - 16, v1)
        unpacked = torch.empty(
            self.out_features, self.in_features, dtype=torch.float32, device=self.qweight.device
        )
        unpacked[..., 0::2] = q0.float()
        unpacked[..., 1::2] = q1.float()

        # 2. Rescale with group scales
        scales_exp = self.scales.float().repeat_interleave(self.group_size, dim=1)
        w_hat = unpacked * scales_exp

        # 3. DV-SSQ: Override salient channels with INT8 precision
        if getattr(self, "salient_weight", None) is not None and self.salient_weight.numel() > 0:
            salient_decomp = self.salient_weight.float() * self.salient_scale.float()
            w_hat[:, self.salient_idx.long()] = salient_decomp

        # 4. Add low-rank residual: A @ B
        w_rot = w_hat + (self.res_u.float() @ self.res_v.float())

        # 5. Un-rotate: W = W_rot @ H
        if self.hadamard and unrotate and (self.in_features % self.hadamard_dim == 0):
            H = self.get_hadamard_matrix(self.qweight.device)
            w_final = (w_rot.view(self.out_features, -1, self.hadamard_dim) @ H).view(
                self.out_features, self.in_features
            )
        else:
            w_final = w_rot

        return w_final.to(self.scales.dtype)

    def materialize(self):
        """Precomputes and caches uncompressed weight in memory for zero-overhead BLAS GEMM."""
        self._cached_weight = self.dequantize(unrotate=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cached_weight is not None:
            return F.linear(x, self._cached_weight, self.bias)
        w = self.dequantize(unrotate=True)
        return F.linear(x, w, self.bias)


# ==============================================================================
# BASELINE SPARK TRANSFORMER PRIMITIVES
# ==============================================================================
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_rope_cos_sin(positions, head_dim, rope_theta, partial_rotary_factor=1.0, device="cpu"):
    rope_head_dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, rope_head_dim, 2, dtype=torch.int64).to(device="cpu", dtype=torch.float) / rope_head_dim)
    )
    inv_freq = inv_freq.to(device)
    t = positions.to(device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    freqs = torch.cat([freqs, freqs], dim=-1)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_pos_emb(x, cos, sin):
    rope_head_dim = cos.shape[-1]
    x_f32 = x.float()
    if x_f32.shape[-1] > rope_head_dim:
        x_rot = x_f32[..., :rope_head_dim]
        x_pass = x_f32[..., rope_head_dim:]
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        x_rot = x_rot * c + rotate_half(x_rot) * s
        result = torch.cat([x_rot, x_pass], dim=-1)
    else:
        c = cos.unsqueeze(0).unsqueeze(0)
        s = sin.unsqueeze(0).unsqueeze(0)
        result = x_f32 * c + rotate_half(x_f32) * s
    return result.to(x.dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    if scaling is None:
        scaling = 1.0 / math.sqrt(query.shape[-1])

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # Key-Value Binding Softmax Sharpening (KV-BSS) & Attention Haze Suppression
    if getattr(module, "key_value_binding_sharpening", False):
        max_logits = attn_weights.max(dim=-1, keepdim=True).values
        haze_mask = attn_weights < (max_logits - 12.0)
        attn_weights = attn_weights.masked_fill(haze_mask, float('-inf'))

    attn_weights = attn_weights - attn_weights.max(dim=-1, keepdim=True).values
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    return attn_output, attn_weights


class Spark2_5RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.float() * hidden_states).to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(Spark2_5RMSNorm)


class Spark2_5MLP(nn.Module):
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.layer_idx = layer_idx

        use_quant = getattr(config, "gsq_int4", False)
        if use_quant:
            quant_config = getattr(config, "fquant_quantization_config", None)
            if not isinstance(quant_config, dict):
                # Backward compatibility for checkpoints with the old field.
                quant_config = getattr(config, "quantization_config", {})
            bifurcation_layers = quant_config.get("bifurcation_layers", [])
            is_bifurcation = layer_idx in bifurcation_layers
            rank = 32 if is_bifurcation else 16
            self.gate_proj = HadamardGSQLinear(
                self.hidden_size, self.intermediate_size, rank=rank, bias=config.mlp_bias
            )
            self.up_proj = HadamardGSQLinear(
                self.hidden_size, self.intermediate_size, rank=rank, bias=config.mlp_bias
            )
            self.down_proj = HadamardGSQLinear(
                self.intermediate_size, self.hidden_size, rank=rank, bias=config.mlp_bias
            )
        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)

        if config.hidden_act != "gelu":
            raise ValueError(f"Only hidden_act='gelu' supported, received: {config.hidden_act}")

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Spark2_5Attention(nn.Module):
    def __init__(self, config: Spark2_5Config, layer_idx: int | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx if layer_idx is not None else 0
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Key-Value Binding Softmax Sharpening (KV-BSS) & Contrastive Focus Scaling
        kv_focus = getattr(config, "kv_focus_factor", 1.0)
        use_kv_sharpening = getattr(config, "key_value_binding_sharpening", False)
        if use_kv_sharpening:
            self.scaling = float(kv_focus) / math.sqrt(self.head_dim)
        else:
            self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.key_value_binding_sharpening = use_kv_sharpening
        self.kv_focus_factor = kv_focus

        self.headwise_attn_output_gate = config.headwise_attn_output_gate
        self.gate_attn_act_mode = config.gate_attn_act_mode
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim

        # Selective Attention Preservation: 100% uncompressed BF16 pass-through
        qkv_out_dim = self.q_dim + 2 * self.kv_dim
        self.q_k_v_proj = nn.Linear(self.hidden_size, qkv_out_dim, bias=config.attention_bias)
        self.g_proj = (
            nn.Linear(self.hidden_size, self.num_heads, bias=config.attention_bias)
            if self.headwise_attn_output_gate
            else None
        )
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self.sliding_window = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        bsz, seq_len = input_shape

        qkv = self.q_k_v_proj(hidden_states)
        q = qkv[..., : self.q_dim]
        k = qkv[..., self.q_dim : self.q_dim + self.kv_dim]
        v = qkv[..., self.q_dim + self.kv_dim :]
        gate_score = self.g_proj(hidden_states) if self.g_proj is not None else None

        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if gate_score is not None:
            gate_score = gate_score.view(bsz, seq_len, self.num_heads, 1).transpose(1, 2)

        cos, sin = position_embeddings
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        attn_output, attn_weights = eager_attention_forward(
            self,
            q,
            k,
            v,
            attention_mask=attention_mask,
            scaling=self.scaling,
            dropout=self.attention_dropout if self.training else 0.0,
        )

        if gate_score is not None:
            if self.gate_attn_act_mode == "sigmoid":
                gate = torch.sigmoid(gate_score.float())
            elif self.gate_attn_act_mode == "silu":
                gate = F.silu(gate_score.float())
            else:
                raise ValueError(f"Unsupported gate_attn_act_mode: {self.gate_attn_act_mode}")
            gate = gate.to(attn_output.dtype)
            attn_output = attn_output * gate

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class Spark2_5DecoderLayer(nn.Module):
    def __init__(self, config: Spark2_5Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Spark2_5Attention(config=config, layer_idx=layer_idx)
        self.mlp = Spark2_5MLP(config, layer_idx=layer_idx)
        self.input_layernorm = Spark2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Spark2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.layer_type = (
            config.layer_types[layer_idx] if layer_idx < len(config.layer_types) else "full_attention"
        )
        if self.layer_type == "sliding_attention" and config.sliding_window is not None:
            self.self_attn.sliding_window = config.sliding_window
        else:
            self.self_attn.sliding_window = None
        self.self_attn.partial_rotary_factor = config.get_partial_rotary_factor(self.layer_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_dtype = self.self_attn.q_k_v_proj.weight.dtype
        hidden_states = hidden_states.to(attn_dtype)

        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states.to(attn_dtype)

        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Spark2_5PreTrainedModel(PreTrainedModel):
    config_class = Spark2_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Spark2_5DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Spark2_5Model(Spark2_5PreTrainedModel):
    def __init__(self, config: Spark2_5Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [Spark2_5DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = Spark2_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in config.layer_types

        self.post_init()

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def materialize_weights(self):
        """Precomputes and caches uncompressed weights across all layers for fast BLAS inference."""
        for module in self.modules():
            if isinstance(module, HadamardGSQLinear):
                module.materialize()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # NOTE (transformers>=5 compat): generate() may pass trimmed inputs_embeds
        # with FULL-length position_ids / stale cache_position. Rebuild whenever
        # shapes disagree with the current input length, else RoPE broadcast can
        # silently expand the query length and break attention masking.
        _q_len = inputs_embeds.shape[1]
        if cache_position is None or cache_position.shape[0] != _q_len:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + _q_len, device=inputs_embeds.device
            )
        if position_ids is None or position_ids.shape[-1] != _q_len:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
        else:
            causal_mask_mapping = attention_mask

        hidden_states = inputs_embeds.float()
        device = hidden_states.device
        dtype = self.embedding.weight.dtype

        head_dim = self.config.head_dim
        rope_cache = {}
        for lt in set(self.config.layer_types):
            rope_theta = self.config.get_rope_theta(lt)
            prf = self.config.get_partial_rotary_factor(lt)
            cos, sin = compute_rope_cos_sin(
                cache_position, head_dim, rope_theta, partial_rotary_factor=prf, device=device
            )
            rope_cache[lt] = (cos, sin)

        for decoder_layer in self.layers:
            layer_type = decoder_layer.layer_type
            position_embeddings = rope_cache.get(layer_type, rope_cache.get("full_attention"))
            layer_attention_mask = causal_mask_mapping.get(layer_type, causal_mask_mapping.get("full_attention"))

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    position_embeddings,
                    layer_attention_mask,
                )
                hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=layer_attention_mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    position_ids=position_ids,
                )

        hidden_states = self.norm(hidden_states)
        hidden_states = hidden_states.to(dtype)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Spark2_5ForCausalLM(Spark2_5PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embedding.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Spark2_5Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding

    def set_input_embeddings(self, value):
        self.model.embedding = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def materialize_weights(self):
        """Materializes all quantized linear layers for maximum inference speed."""
        self.model.materialize_weights()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int = 0,
        token_type_ids: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]

        if self.config.tie_word_embeddings:
            embed_weight = self.model.embedding.weight
            logits = F.linear(hidden_states, embed_weight)
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["Spark2_5Config", "Spark2_5ForCausalLM", "Spark2_5Model", "HadamardGSQLinear"]
