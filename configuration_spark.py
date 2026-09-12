# coding=utf-8
# Copyright 2026 The XHToken team and the HuggingFace Inc. team. All rights reserved.
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

from transformers import PretrainedConfig


class Spark2_5Config(PretrainedConfig):
    model_type = "spark2_5"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_k_v_proj": "colwise",
        "layers.*.self_attn.g_proj": "colwise",
        "layers.*.self_attn.out_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embedding": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=131072,
        hidden_size=2560,
        intermediate_size=10240,
        num_hidden_layers=36,
        num_attention_heads=16,
        num_key_value_heads=4,
        hidden_act="gelu",
        max_position_embeddings=1048576,
        initializer_range=0.01976,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=2,
        bos_token_id=0,
        eos_token_id=1,
        tie_word_embeddings=True,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=256,
        headwise_attn_output_gate=True,
        gate_attn_act_mode="sigmoid",
        sliding_window=512,
        layer_types=([
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]),
        gsq_int4=False,
        hadamard_spin=False,
        dv_ssq=False,
        selective_attention_preservation=False,
        key_value_binding_sharpening=False,
        kv_focus_factor=1.10,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})"
            )
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        self.headwise_attn_output_gate = headwise_attn_output_gate
        self.gate_attn_act_mode = gate_attn_act_mode
        self.sliding_window = sliding_window
        self.gsq_int4 = gsq_int4
        self.hadamard_spin = hadamard_spin
        self.dv_ssq = dv_ssq
        self.selective_attention_preservation = selective_attention_preservation
        self.key_value_binding_sharpening = key_value_binding_sharpening
        self.kv_focus_factor = kv_focus_factor
        if layer_types is None:
            layer_types = ["full_attention"] * num_hidden_layers
        if len(layer_types) != num_hidden_layers:
            raise ValueError(
                f"layer_types length ({len(layer_types)}) must match num_hidden_layers ({num_hidden_layers})"
            )
        self.layer_types = layer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            gsq_int4=gsq_int4,
            hadamard_spin=hadamard_spin,
            dv_ssq=dv_ssq,
            selective_attention_preservation=selective_attention_preservation,
            key_value_binding_sharpening=key_value_binding_sharpening,
            kv_focus_factor=kv_focus_factor,
            **kwargs,
        )
        # Spark uses a per-layer RoPE schema consumed by this custom modeling
        # implementation. Set it after generic config initialization so
        # Transformers does not reinterpret the nested mapping.
        self.rope_parameters = rope_parameters or {}

    def get_rope_theta(self, layer_type):
        rope_parameters = self.rope_parameters or {}
        params = rope_parameters.get(layer_type, rope_parameters if "rope_theta" in rope_parameters else {})
        return params.get("rope_theta", 10000)

    def get_partial_rotary_factor(self, layer_type):
        rope_parameters = self.rope_parameters or {}
        params = rope_parameters.get(layer_type, rope_parameters if "partial_rotary_factor" in rope_parameters else {})
        return params.get("partial_rotary_factor", 1.0)


__all__ = ["Spark2_5Config"]
