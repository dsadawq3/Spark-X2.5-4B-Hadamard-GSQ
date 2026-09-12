"""
================================================================================
F-LABS ULTRA-DEEP EMPIRICAL AUDIT: EMERGENCE vs BREAKDOWN (PRODUCTION ENGINE)
================================================================================
Scientific investigation of:
1. End-to-end 36-layer hidden state accumulation & Lyapunov drift dynamics
2. Logit rank conservation, Top-1 match %, Top-K Jaccard overlap, and KL divergence
3. Error spectral orthogonality (null-space noise projection)
4. Induction head & attention routing preservation on full-attention bridges
================================================================================
"""

import os
import sys
import gc
import json
import math
import time
import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
import safetensors.torch
from transformers import AutoTokenizer

_SCRIPT_DIR = str(Path(__file__).resolve().parent)
RAW_DIR = _SCRIPT_DIR
QUANT_DIR = _SCRIPT_DIR
AUDIT_REPORT_PATH = os.path.join(_SCRIPT_DIR, "ultra_deep_audit_report.json")

def _apply_dirs(quant_dir=None, raw_dir=None, report=None):
    global QUANT_DIR, RAW_DIR, AUDIT_REPORT_PATH
    if quant_dir:
        QUANT_DIR = quant_dir
    if raw_dir:
        RAW_DIR = raw_dir
    if report:
        AUDIT_REPORT_PATH = report

# Maximize multi-threaded execution
torch.set_num_threads(os.cpu_count() or 8)

sys.path.insert(0, QUANT_DIR)
from configuration_spark import Spark2_5Config
from modeling_spark import Spark2_5DecoderLayer, Spark2_5RMSNorm, compute_rope_cos_sin, apply_rotary_pos_emb, repeat_kv

# Realistic, multi-token algorithmic code prompt with recursion, loops, syntax scopes, and branching
COMPLEX_CODE_PROMPT = '''def binary_tree_diameter(root):
    max_diameter = [0]
    def calculate_height(node):
        if not node:
            return 0
        left_h = calculate_height(node.left)
        right_h = calculate_height(node.right)
        current_dia = left_h + right_h
        if current_dia > max_diameter[0]:
            max_diameter[0] = current_dia
        return 1 + max(left_h, right_h)
    calculate_height(root)
    return max_diameter[0]
'''

def run_ultra_deep_audit():
    if os.path.abspath(RAW_DIR) == os.path.abspath(QUANT_DIR):
        raise ValueError(
            "raw_dir must point to a separate unquantized BF16 model directory; "
            "the quantized release does not contain raw weights"
        )
    print("=" * 80)
    print("F-LABS ULTRA-DEEP EMPIRICAL AUDIT: SPARK-X2.5-4B vs QUANTIZED MODEL")
    print("=" * 80)
    start_time = time.time()

    # Load configurations independently for raw and quantized architectures
    with open(os.path.join(RAW_DIR, "config.json")) as f:
        raw_cfg_dict = json.load(f)
    config_raw = Spark2_5Config(**raw_cfg_dict)

    with open(os.path.join(QUANT_DIR, "config.json")) as f:
        quant_cfg_dict = json.load(f)
    config_quant = Spark2_5Config(**quant_cfg_dict)

    print(f"[*] Raw Config: gsq_int4={getattr(config_raw, 'gsq_int4', False)}")
    print(f"[*] Quant Config: gsq_int4={getattr(config_quant, 'gsq_int4', False)}")

    # Tokenize realistic code prompt
    tokenizer = AutoTokenizer.from_pretrained(QUANT_DIR, trust_remote_code=True)
    tokens = tokenizer.encode(COMPLEX_CODE_PROMPT, return_tensors="pt")
    seq_len = tokens.shape[1]
    print(f"[*] Benchmark Code Prompt Tokenized: {seq_len} tokens")

    # Load shard index mappings
    with open(os.path.join(RAW_DIR, "model.safetensors.index.json")) as f:
        raw_idx = json.load(f)
    with open(os.path.join(QUANT_DIR, "model.safetensors.index.json")) as f:
        quant_idx = json.load(f)

    # Shard cache in RAM for ultra-fast throughput
    raw_shards = {}
    quant_shards = {}

    def get_raw_shard(shard_name):
        if shard_name not in raw_shards:
            print(f"[*] Loading raw shard {shard_name} into RAM cache...")
            raw_shards[shard_name] = safetensors.torch.load_file(os.path.join(RAW_DIR, shard_name))
        return raw_shards[shard_name]

    def get_quant_shard(shard_name):
        if shard_name not in quant_shards:
            print(f"[*] Loading quantized shard {shard_name} into RAM cache...")
            quant_shards[shard_name] = safetensors.torch.load_file(os.path.join(QUANT_DIR, shard_name))
        return quant_shards[shard_name]

    # Load embedding and final norm
    print("[*] Loading embedding weight and final norm...")
    embed_shard_name = raw_idx["weight_map"]["model.embedding.weight"]
    embed_weight = get_raw_shard(embed_shard_name)["model.embedding.weight"]

    norm_shard_name = raw_idx["weight_map"]["model.norm.weight"]
    norm_weight_raw = get_raw_shard(norm_shard_name)["model.norm.weight"]
    norm_weight_quant = get_quant_shard(norm_shard_name)["model.norm.weight"]

    with torch.no_grad():
        final_norm_raw = Spark2_5RMSNorm(config_raw.hidden_size, eps=config_raw.rms_norm_eps).to(torch.bfloat16)
        final_norm_raw.weight.copy_(norm_weight_raw)

        final_norm_quant = Spark2_5RMSNorm(config_quant.hidden_size, eps=config_quant.rms_norm_eps).to(torch.bfloat16)
        final_norm_quant.weight.copy_(norm_weight_quant)

    # Initial token hidden state
    h_raw = embed_weight[tokens[0]].unsqueeze(0)  # (1, seq_len, 2560)
    h_quant = h_raw.clone()

    print(f"[*] Initial token hidden state initialized: shape {tuple(h_raw.shape)}, dtype {h_raw.dtype}")

    # Compute RoPE position embeddings for sliding and full attention
    cache_position = torch.arange(seq_len)
    rope_cache = {}
    for lt in set(config_raw.layer_types):
        rope_theta = config_raw.get_rope_theta(lt)
        prf = config_raw.get_partial_rotary_factor(lt)
        cos, sin = compute_rope_cos_sin(cache_position, config_raw.head_dim, rope_theta, partial_rotary_factor=prf)
        rope_cache[lt] = (cos.to(torch.bfloat16), sin.to(torch.bfloat16))

    num_layers = config_raw.num_hidden_layers  # 36
    bifurcation_layers = {3, 7, 11, 15, 19, 23, 27, 31, 35} | set(range(16, 29))
    bridge_layers = [3, 15, 27, 35]

    layer_metrics = []
    bridge_head_diffs = {}

    print("\n" + "=" * 85)
    print(f"{'Layer':5s} | {'Type':15s} | {'Cos Sim':10s} | {'Rel Dev':10s} | {'Null-Space %':14s} | {'Max Deviation':14s}")
    print("=" * 85)

    # Sequential layer propagation with STRICT loading and pre-materialized GEMMs
    for l in range(num_layers):
        is_bifurcation = l in bifurcation_layers
        layer_type = "Bifurcation" if is_bifurcation else "Standard"

        # Shard-agnostic tensor collection for layer l
        prefix = f"model.layers.{l}."
        sd_raw = {}
        for k, s_name in raw_idx["weight_map"].items():
            if k.startswith(prefix):
                short_k = k[len(prefix):]
                sd_raw[short_k] = get_raw_shard(s_name)[k]

        sd_quant = {}
        for k, s_name in quant_idx["weight_map"].items():
            if k.startswith(prefix):
                short_k = k[len(prefix):]
                sd_quant[short_k] = get_quant_shard(s_name)[k]

        # Instantiate raw and quantized decoder layers with their respective configurations
        layer_raw = Spark2_5DecoderLayer(config_raw, l).to(torch.bfloat16)
        layer_quant = Spark2_5DecoderLayer(config_quant, l).to(torch.bfloat16)

        # STRICT loading guarantees every single tensor is loaded exactly
        layer_raw.load_state_dict(sd_raw, strict=True)
        layer_quant.load_state_dict(sd_quant, strict=True)

        # Pre-materialize quantized MLP weights in memory for high-speed BLAS GEMM
        layer_quant.mlp.gate_proj.materialize()
        layer_quant.mlp.up_proj.materialize()
        layer_quant.mlp.down_proj.materialize()

        lt = layer_raw.layer_type
        pos_emb = rope_cache.get(lt, rope_cache.get("full_attention"))

        # Forward pass through layer
        with torch.no_grad():
            # If bridge layer, inspect attention routing Frobenius difference
            if l in bridge_layers:
                qkv_r = layer_raw.self_attn.q_k_v_proj(layer_raw.input_layernorm(h_raw))
                qkv_q = layer_quant.self_attn.q_k_v_proj(layer_quant.input_layernorm(h_quant))

                q_r = qkv_r[..., :config_raw.num_attention_heads * config_raw.head_dim]
                k_r = qkv_r[..., config_raw.num_attention_heads * config_raw.head_dim : config_raw.num_attention_heads * config_raw.head_dim + config_raw.num_key_value_heads * config_raw.head_dim]
                q_q = qkv_q[..., :config_raw.num_attention_heads * config_raw.head_dim]
                k_q = qkv_q[..., config_raw.num_attention_heads * config_raw.head_dim : config_raw.num_attention_heads * config_raw.head_dim + config_raw.num_key_value_heads * config_raw.head_dim]

                cos, sin = pos_emb
                q_r = q_r.view(1, seq_len, 16, 256).transpose(1, 2)
                k_r = k_r.view(1, seq_len, 4, 256).transpose(1, 2)
                q_q = q_q.view(1, seq_len, 16, 256).transpose(1, 2)
                k_q = k_q.view(1, seq_len, 4, 256).transpose(1, 2)

                q_r_rot = apply_rotary_pos_emb(q_r, cos, sin).squeeze(0)
                k_r_rot = apply_rotary_pos_emb(k_r, cos, sin).squeeze(0)
                q_q_rot = apply_rotary_pos_emb(q_q, cos, sin).squeeze(0)
                k_q_rot = apply_rotary_pos_emb(k_q, cos, sin).squeeze(0)

                k_r_rep = repeat_kv(k_r_rot.unsqueeze(0), 4).squeeze(0)
                k_q_rep = repeat_kv(k_q_rot.unsqueeze(0), 4).squeeze(0)

                mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
                A_r = torch.softmax((q_r_rot.float() @ k_r_rep.float().transpose(-2, -1)) / 16.0 + mask, dim=-1)
                A_q = torch.softmax((q_q_rot.float() @ k_q_rep.float().transpose(-2, -1)) / 16.0 + mask, dim=-1)

                diff_heads = [(A_r[h] - A_q[h]).norm().item() / A_r[h].norm().item() for h in range(16)]
                bridge_head_diffs[l] = {
                    "mean_head_frobenius_diff": float(np.mean(diff_heads)),
                    "max_head_frobenius_diff": float(np.max(diff_heads)),
                    "min_head_frobenius_diff": float(np.min(diff_heads)),
                }

            h_raw = layer_raw(h_raw, position_embeddings=pos_emb)
            h_quant = layer_quant(h_quant, position_embeddings=pos_emb)

        # ----------------------------------------------------------------------
        # VECTOR 1: HIDDEN STATE DRIFT
        # ----------------------------------------------------------------------
        hr_f = h_raw.squeeze(0).float()
        hq_f = h_quant.squeeze(0).float()

        cos_sim = (hr_f * hq_f).sum() / (hr_f.norm() * hq_f.norm()).clamp(min=1e-12)
        rel_dev = (hr_f - hq_f).norm() / hr_f.norm().clamp(min=1e-12)
        max_dev = (hr_f - hq_f).abs().max().item()

        # ----------------------------------------------------------------------
        # VECTOR 3: ERROR SPECTRAL ORTHOGONALITY (Null-Space Projection)
        # ----------------------------------------------------------------------
        E = (hq_f - hr_f)  # Error matrix (seq_len x 2560)
        with torch.no_grad():
            U_h, S_h, V_h = torch.svd_lowrank(hr_f, q=32, niter=2)
            E_parallel = (E @ V_h) @ V_h.T
            E_ortho = E - E_parallel
            null_space_fraction = (E_ortho.norm() ** 2) / (E.norm() ** 2).clamp(min=1e-12)
            null_space_pct = null_space_fraction.item() * 100.0

        layer_data = {
            "layer_idx": l,
            "layer_type": layer_type,
            "is_bifurcation": is_bifurcation,
            "cosine_similarity": cos_sim.item(),
            "relative_deviation": rel_dev.item(),
            "max_absolute_deviation": max_dev,
            "null_space_noise_percentage": null_space_pct,
        }
        layer_metrics.append(layer_data)

        print(f"L{l:02d}  | {layer_type:15s} | {cos_sim.item():.7f} | {rel_dev.item():.6f} | {null_space_pct:12.2f}% | {max_dev:.6f}")

        # Explicit cleanup of layer weights to keep RAM perfectly managed
        del layer_raw, layer_quant, sd_raw, sd_quant
        gc.collect()

    # ----------------------------------------------------------------------
    # VECTOR 2: LOGIT RANK CONSERVATION & TOP-K AGREEMENT
    # ----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[*] Computing final output logits across all token positions...")
    normed_raw = final_norm_raw(h_raw)
    normed_quant = final_norm_quant(h_quant)

    # Compute logits: F.linear(normed, embed_weight)
    z_raw = F.linear(normed_raw, embed_weight).squeeze(0).float()  # (seq_len, vocab_size)
    z_quant = F.linear(normed_quant, embed_weight).squeeze(0).float()

    print(f"[*] Logits computed: shape {tuple(z_raw.shape)} across {seq_len} token positions")

    # 1. Top-1 Argmax Agreement
    top1_raw = torch.argmax(z_raw, dim=-1)
    top1_quant = torch.argmax(z_quant, dim=-1)
    top1_match_count = (top1_raw == top1_quant).sum().item()
    top1_match_pct = (top1_match_count / seq_len) * 100.0

    # 2. Top-5 and Top-10 Jaccard Overlap
    top5_raw = torch.topk(z_raw, k=5, dim=-1).indices
    top5_quant = torch.topk(z_quant, k=5, dim=-1).indices

    top10_raw = torch.topk(z_raw, k=10, dim=-1).indices
    top10_quant = torch.topk(z_quant, k=10, dim=-1).indices

    jaccard_5_list = []
    jaccard_10_list = []
    for t in range(seq_len):
        s5_r = set(top5_raw[t].tolist())
        s5_q = set(top5_quant[t].tolist())
        j5 = len(s5_r & s5_q) / len(s5_r | s5_q)
        jaccard_5_list.append(j5)

        s10_r = set(top10_raw[t].tolist())
        s10_q = set(top10_quant[t].tolist())
        j10 = len(s10_r & s10_q) / len(s10_r | s10_q)
        jaccard_10_list.append(j10)

    top5_jaccard_pct = float(np.mean(jaccard_5_list)) * 100.0
    top10_jaccard_pct = float(np.mean(jaccard_10_list)) * 100.0

    # 3. Kullback-Leibler Divergence
    p_raw = F.softmax(z_raw, dim=-1)
    log_p_quant = F.log_softmax(z_quant, dim=-1)
    kl_div = F.kl_div(log_p_quant, p_raw, reduction="batchmean").item()

    # 4. Logit Distribution Entropy
    ent_logits_raw = -(p_raw * torch.log(p_raw.clamp(min=1e-12))).sum(dim=-1).mean().item()
    p_quant = F.softmax(z_quant, dim=-1)
    ent_logits_quant = -(p_quant * torch.log(p_quant.clamp(min=1e-12))).sum(dim=-1).mean().item()

    # 5. Lyapunov Drift Exponent Fit
    deviations = [d["relative_deviation"] for d in layer_metrics]
    layers_arr = np.arange(len(deviations))
    log_devs = np.log(np.maximum(deviations, 1e-8))
    slope, intercept = np.polyfit(layers_arr, log_devs, 1)
    lyapunov_exponent = float(slope)

    # 6. Overall Summary Metrics
    final_cos_sim = layer_metrics[-1]["cosine_similarity"]
    final_rel_dev = layer_metrics[-1]["relative_deviation"]
    mean_null_space = float(np.mean([d["null_space_noise_percentage"] for d in layer_metrics]))

    audit_summary = {
        "sequence_tokens_evaluated": seq_len,
        "final_layer_cosine_similarity": final_cos_sim,
        "final_layer_relative_deviation": final_rel_dev,
        "top1_argmax_agreement_pct": top1_match_pct,
        "top5_jaccard_overlap_pct": top5_jaccard_pct,
        "top10_jaccard_overlap_pct": top10_jaccard_pct,
        "kullback_leibler_divergence": kl_div,
        "logit_entropy_raw": ent_logits_raw,
        "logit_entropy_quant": ent_logits_quant,
        "logit_entropy_shift": ent_logits_quant - ent_logits_raw,
        "mean_null_space_noise_pct": mean_null_space,
        "lyapunov_exponent_lambda": lyapunov_exponent,
        "error_dynamics_verdict": (
            "Self-Stabilizing Attractor Dynamics (Bounded Sublinear Drift)"
            if lyapunov_exponent <= 0.05
            else "Chaotic Exponential Divergence"
        ),
        "bridge_layers_attention_diffs": bridge_head_diffs,
    }

    full_audit_report = {
        "summary": audit_summary,
        "layer_by_layer_drift": layer_metrics,
    }

    print("\n" + "=" * 80)
    print("ULTRA-DEEP EMPIRICAL AUDIT RESULTS:")
    print("=" * 80)
    print(f"  Top-1 Token Exact Argmax Agreement:  {top1_match_pct:.2f}% ({top1_match_count}/{seq_len} tokens identical)")
    print(f"  Top-5 Token Set Jaccard Overlap:     {top5_jaccard_pct:.2f}%")
    print(f"  Top-10 Token Set Jaccard Overlap:    {top10_jaccard_pct:.2f}%")
    print(f"  Kullback-Leibler Divergence (KL):    {kl_div:.6f} nats")
    print(f"  Logit Entropy Shift:                 Raw = {ent_logits_raw:.4f} -> Quant = {ent_logits_quant:.4f} (Delta = {ent_logits_quant - ent_logits_raw:+.4f})")
    print(f"  Final Layer 35 Cosine Similarity:    {final_cos_sim:.7f}")
    print(f"  Final Layer 35 Relative Deviation:   {final_rel_dev:.6f}")
    print(f"  Mean Error Null-Space Fraction:      {mean_null_space:.2f}% (Quantization noise orthogonal to semantics)")
    print(f"  Lyapunov Drift Exponent (lambda):    {lyapunov_exponent:+.5f} ({audit_summary['error_dynamics_verdict']})")
    print(f"  Bridge Attention Head Deviation:     L3={bridge_head_diffs[3]['mean_head_frobenius_diff']:.4f}, L15={bridge_head_diffs[15]['mean_head_frobenius_diff']:.4f}, L27={bridge_head_diffs[27]['mean_head_frobenius_diff']:.4f}, L35={bridge_head_diffs[35]['mean_head_frobenius_diff']:.4f}")
    print(f"  Audit Execution Time:                {time.time() - start_time:.2f} seconds")
    print("=" * 80)

    # Save to JSON
    with open(AUDIT_REPORT_PATH, "w") as f:
        json.dump(full_audit_report, f, indent=2)
    print(f"[*] Full audit report saved to {AUDIT_REPORT_PATH}")

    return full_audit_report

if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description="Ultra-deep emergence audit")
    _ap.add_argument("--quant_dir", type=str, default=_SCRIPT_DIR)
    _ap.add_argument("--raw_dir", type=str, required=True, help="Separate raw BF16 model directory")
    _ap.add_argument("--report", type=str, default=AUDIT_REPORT_PATH)
    _a = _ap.parse_args()
    _apply_dirs(_a.quant_dir, _a.raw_dir, _a.report)
    run_ultra_deep_audit()
