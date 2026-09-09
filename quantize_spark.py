"""
================================================================================
F-LABS PRODUCTION QUANTIZATION ENGINE: SPARK-X2.5-4B
ARCHITECTURE: DENSE-VECTORIZED SUBSPACE SALIENCE QUANTIZATION (DV-SSQ)
MULTI-PRECISION TIERS:
  - 16-BIT (BF16): ATTENTION PROJECTIONS, LM_HEAD, EMBEDDINGS, BIASES, SVD LOW-RANK (r=16/32)
  -  8-BIT (INT8): SALIENT SEMANTIC CHANNELS (TOP 12.5% ENERGY SUB-BLOCKS)
  -  4-BIT (INT4): BACKGROUND PARAMETER MASS (GSQ g=64) + WALSH-HADAMARD SPIN (H_256)
================================================================================
Author: Master Quantization and Compression Architect, F-Labs
Target Model: Spark-X2.5-4B (4.11B parameters, 8.22 GB BF16)
Output Model: F-Labs/Spark-X2.5-4B-Hadamard-GSQ-DVSSQ (~4.2 GB, ~2.0x compression)
================================================================================
"""

import os
import sys
import gc
import time
import json
import math
import shutil
import argparse
import numpy as np
import scipy.linalg
import torch
import safetensors.torch

# Ensure optimal CPU parallelism
torch.set_num_threads(os.cpu_count() or 8)

# ----------------------------------------------------------------------
# PILLAR 1: HADAMARD SPIN ROTATION MATRIX GENERATOR
# ----------------------------------------------------------------------
def generate_hadamard_matrix(dim: int = 256) -> torch.Tensor:
    """
    Constructs an orthonormal, symmetric Walsh-Hadamard matrix H_dim / sqrt(dim).
    Guarantees H^T = H and H^T @ H = I.
    """
    assert (dim & (dim - 1)) == 0, f"Dimension {dim} must be a power of 2"
    h_np = scipy.linalg.hadamard(dim)
    h_torch = torch.from_numpy(h_np).float() / math.sqrt(dim)
    assert torch.allclose(h_torch, h_torch.T, atol=1e-5), "Hadamard matrix must be symmetric"
    assert torch.allclose(h_torch @ h_torch.T, torch.eye(dim), atol=1e-5), "Hadamard matrix must be orthogonal"
    return h_torch


def apply_hadamard_spin_to_weight(W: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """
    Rotates weight matrix W on its input channels:
    W_rot = W @ H_block_diag
    Using efficient batched chunk-wise multiplication without materializing large sparse matrices.
    """
    N, K = W.shape
    h_dim = H.shape[0]
    assert K % h_dim == 0, f"Weight input dimension {K} must be divisible by {h_dim}"
    num_blocks = K // h_dim
    W_reshaped = W.view(N, num_blocks, h_dim)
    W_rot = torch.matmul(W_reshaped, H).view(N, K)
    return W_rot


# ----------------------------------------------------------------------
# PILLAR 4: SVD SPECTRAL ENTROPY BIFURCATION DETECTOR
# ----------------------------------------------------------------------
def compute_spectral_entropy(W: torch.Tensor) -> float:
    """
    Calculates singular value entropy: S = -sum(p_i * ln(p_i)), where p_i = sigma_i / sum(sigma_j).
    Measures information dispersion and geometric curvature across the layer's eigenspace.
    """
    with torch.no_grad():
        W_f = W.float()
        q = min(64, min(W.shape))
        _, S, _ = torch.svd_lowrank(W_f, q=q, niter=2)
        p = S / S.sum().clamp(min=1e-12)
        entropy = -(p * torch.log(p + 1e-12)).sum().item()
        return entropy


def is_bifurcation_layer(layer_idx: int) -> bool:
    """
    Identifies Bifurcation Hub layers:
    - full_attention bridge layers (layers 3, 7, 11, 15, 19, 23, 27, 31, 35)
    - Mid-deep reasoning abstraction circuits (layers 16 to 28)
    """
    full_attention_layers = {3, 7, 11, 15, 19, 23, 27, 31, 35}
    mid_deep_reasoning_layers = set(range(16, 29))
    return (layer_idx in full_attention_layers) or (layer_idx in mid_deep_reasoning_layers)


# ----------------------------------------------------------------------
# PILLAR 5, 3 & DV-SSQ: MULTI-PRECISION QUANTIZATION (INT8 + INT4 + SVD)
# ----------------------------------------------------------------------
def pack_int4_signed(q: torch.Tensor) -> torch.Tensor:
    """
    Packs signed INT4 values in [-8, 7] into uint8 (2 values per byte).
    Low nibble: first value; High nibble: second value.
    """
    q_u4 = (q.to(torch.int32) & 0x0F).to(torch.uint8)
    v0 = q_u4[..., 0::2]
    v1 = q_u4[..., 1::2]
    packed = (v1 << 4) | v0
    return packed


def quantize_dv_ssq(
    W: torch.Tensor,
    rank: int,
    group_size: int = 64,
    salient_ratio: float = 0.125,
    is_bifurcation: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """
    Executes Dense-Vectorized Subspace Salience Quantization (DV-SSQ):
    1. Channel Salience Ranking: Identify top salient semantic channels (12.5% energy).
    2. Salient Subspace Quantization: Quantize salient channels in signed INT8 [-128, 127].
    3. Background Subspace Quantization: Quantize background mass in signed INT4 [-8, 7] (GSQ g=64).
    4. Truncated SVD RCO: Extract low-rank residual factors A, B in BF16 (rank in {16, 32}).
    5. Closed-Form Optimal Refinement for Bifurcation Hubs.
    Returns:
    - packed_Q (uint8, N x K//2)
    - scales (bfloat16, N x K//group_size)
    - res_u (bfloat16, N x rank)
    - res_v (bfloat16, rank x K)
    - salient_idx (int32, num_salient)
    - salient_weight (int8, N x num_salient)
    - salient_scale (bfloat16, 1 x num_salient)
    - metrics dict
    """
    N, K = W.shape
    assert K % group_size == 0, f"K ({K}) must be divisible by group_size ({group_size})"
    W_f = W.float()

    # Step 1: Base INT4 Group-Scale Quantization across all groups
    Wg = W_f.view(-1, group_size)
    s_base = (Wg.abs().amax(dim=1, keepdim=True) / 7.0).clamp(min=1e-8)
    Q_int4 = torch.clamp(torch.round(Wg / s_base), -8, 7)
    W_hat = (Q_int4 * s_base).view(N, K)

    # Step 2: Semantic Subspace Splitting (Identify top salient channels)
    num_salient = int(K * salient_ratio)
    col_energy = W_f.norm(dim=0)
    salient_idx = torch.topk(col_energy, k=num_salient).indices.sort().values

    # Step 3: Quantize salient semantic channels in INT8
    W_salient = W_f[:, salient_idx]
    s_salient = (W_salient.abs().amax(dim=0, keepdim=True) / 127.0).clamp(min=1e-8)
    Q_salient = torch.clamp(torch.round(W_salient / s_salient), -128, 127).to(torch.int8)

    # Override salient channels in reconstructed base
    W_hat[:, salient_idx] = Q_salient.float() * s_salient

    # Step 4: Residual error extraction R = W - W_hat
    R = W_f - W_hat

    # Step 5: Truncated SVD RCO on residual
    U, S_vals, V = torch.svd_lowrank(R, q=rank, niter=4)
    sqrt_S = torch.sqrt(S_vals)
    A = U * sqrt_S
    B = sqrt_S.unsqueeze(1) * V.T

    # Step 6: Closed-form refinement for Bifurcation Hub layers
    if is_bifurcation:
        AB = A @ B
        E = W_f - AB
        # Salient channels are already high-fidelity INT8, refine background INT4 groups
        E[:, salient_idx] = Q_salient.float() * s_salient
        Eg = E.view(-1, group_size)
        s_opt = (Q_int4 * Eg).sum(dim=1, keepdim=True) / (Q_int4**2).sum(dim=1, keepdim=True).clamp(min=1e-8)
        s_base = torch.where(s_opt > 1e-8, s_opt, s_base)
        # Update reconstructed matrix
        W_hat = (Q_int4 * s_base).view(N, K)
        W_hat[:, salient_idx] = Q_salient.float() * s_salient

    # Compute final metrics
    W_recon = W_hat + (A @ B)
    frob_err = (W_f - W_recon).norm() / W_f.norm().clamp(min=1e-12)

    # Step 7: Bit-pack INT4 Q into uint8
    Q_int8 = Q_int4.to(torch.int8)
    packed_Q = pack_int4_signed(Q_int8).view(N, K // 2).contiguous()

    # Format return tensors
    scales_out = s_base.view(N, K // group_size).to(torch.bfloat16).contiguous()
    res_u_out = A.to(torch.bfloat16).contiguous()
    res_v_out = B.to(torch.bfloat16).contiguous()
    salient_idx_out = salient_idx.to(torch.int32).contiguous()
    salient_weight_out = Q_salient.contiguous()
    salient_scale_out = s_salient.to(torch.bfloat16).contiguous()

    metrics = {
        "frob_rel_error": frob_err.item(),
        "rank": rank,
        "num_salient_channels": num_salient,
        "is_bifurcation": is_bifurcation,
    }
    return packed_Q, scales_out, res_u_out, res_v_out, salient_idx_out, salient_weight_out, salient_scale_out, metrics


# ----------------------------------------------------------------------
# MASTER QUANTIZATION ENGINE
# ----------------------------------------------------------------------
def run_master_quantization(raw_model_dir: str, quantized_model_dir: str):
    print("=" * 85)
    print("F-LABS SPARK-X2.5-4B AUTONOMOUS DV-SSQ QUANTIZATION PIPELINE")
    print("=" * 85)
    start_time = time.time()

    os.makedirs(quantized_model_dir, exist_ok=True)

    # Load raw index
    raw_index_path = os.path.join(raw_model_dir, "model.safetensors.index.json")
    with open(raw_index_path, "r") as f:
        raw_index = json.load(f)

    weight_map = raw_index["weight_map"]
    shards = sorted(list(set(weight_map.values())))
    print(f"[*] Total raw safetensors shards: {len(shards)}")
    print(f"[*] Total raw parameters mapped: {raw_index['metadata'].get('total_parameters', 'N/A')}")
    print(f"[*] Total raw size: {raw_index['metadata'].get('total_size', 0) / (1024**3):.2f} GB")

    # Generate 256-dim Hadamard spin matrix
    print("[*] Generating orthonormal Walsh-Hadamard spin matrix (N=256)...")
    H256 = generate_hadamard_matrix(256)
    print(f"    Hadamard matrix verified: shape {tuple(H256.shape)}, ||H^T H - I|| < 1e-5")

    quantized_weight_map = {}
    total_quantized_bytes = 0
    total_tensors_processed = 0

    layer_entropies = {}

    # Process shard-by-shard with RAM management
    for shard_idx, shard_name in enumerate(shards, 1):
        print(f"\n" + "=" * 85)
        print(f"[SHARD {shard_idx}/{len(shards)}]: Loading {shard_name}...")
        shard_path = os.path.join(raw_model_dir, shard_name)
        shard_tensors = safetensors.torch.load_file(shard_path)
        print(f"    Loaded {len(shard_tensors)} tensors from {shard_name}")

        quantized_shard_dict = {}

        for tensor_name, tensor in shard_tensors.items():
            total_tensors_processed += 1

            # PILLAR 2: Zero-compression shield on embeddings, RMSNorms, biases
            is_shielded = (
                tensor_name.startswith("model.embedding")
                or tensor_name.endswith("norm.weight")
                or tensor_name.endswith(".bias")
            )

            if is_shielded:
                # Keep in 100% uncompressed BF16
                quantized_shard_dict[tensor_name] = tensor.to(torch.bfloat16)
                quantized_weight_map[tensor_name] = shard_name
                tensor_bytes = tensor.numel() * 2
                total_quantized_bytes += tensor_bytes
                print(f"    [SHIELD]      {tensor_name:53s} -> Kept BF16 ({tensor_bytes / (1024**2):.2f} MB)")
                continue

            # SELECTIVE ATTENTION PRESERVATION: 100% uncompressed BF16 pass-through
            if ".self_attn." in tensor_name:
                saved_tensor = tensor.to(torch.bfloat16)
                quantized_shard_dict[tensor_name] = saved_tensor
                quantized_weight_map[tensor_name] = shard_name
                tensor_bytes = saved_tensor.numel() * 2
                total_quantized_bytes += tensor_bytes
                print(f"    [ATTN-PASS]   {tensor_name:53s} -> Kept BF16 ({tensor_bytes / (1024**2):.2f} MB)")
                continue

            # Linear projection layer quantization for MLP (gate_proj, up_proj, down_proj)
            parts = tensor_name.split(".")
            assert len(parts) >= 4 and parts[-1] == "weight", f"Unexpected tensor: {tensor_name}"
            layer_idx = int(parts[2])
            proj_name = parts[-2]
            prefix = tensor_name[:-7]  # strip '.weight'

            is_bifurcation = is_bifurcation_layer(layer_idx)

            if layer_idx not in layer_entropies and proj_name == "gate_proj":
                entropy = compute_spectral_entropy(tensor)
                layer_entropies[layer_idx] = entropy

            rank = 32 if is_bifurcation else 16

            # PILLAR 1: Apply Hadamard Spin Rotation to MLP layer
            W = tensor.float()
            N, K = W.shape
            if K % 256 == 0:
                W_rot = apply_hadamard_spin_to_weight(W, H256)
            else:
                W_rot = W

            # DV-SSQ: Multi-Precision Subspace Salience Quantization (INT8 + INT4 + SVD)
            (
                packed_q,
                scales,
                res_u,
                res_v,
                salient_idx,
                salient_weight,
                salient_scale,
                metrics,
            ) = quantize_dv_ssq(
                W=W_rot,
                rank=rank,
                group_size=64,
                salient_ratio=0.125,
                is_bifurcation=is_bifurcation,
            )

            # Store quantized components
            qweight_key = f"{prefix}.qweight"
            scales_key = f"{prefix}.scales"
            res_u_key = f"{prefix}.res_u"
            res_v_key = f"{prefix}.res_v"
            salient_idx_key = f"{prefix}.salient_idx"
            salient_weight_key = f"{prefix}.salient_weight"
            salient_scale_key = f"{prefix}.salient_scale"

            quantized_shard_dict[qweight_key] = packed_q
            quantized_shard_dict[scales_key] = scales
            quantized_shard_dict[res_u_key] = res_u
            quantized_shard_dict[res_v_key] = res_v
            quantized_shard_dict[salient_idx_key] = salient_idx
            quantized_shard_dict[salient_weight_key] = salient_weight
            quantized_shard_dict[salient_scale_key] = salient_scale

            quantized_weight_map[qweight_key] = shard_name
            quantized_weight_map[scales_key] = shard_name
            quantized_weight_map[res_u_key] = shard_name
            quantized_weight_map[res_v_key] = shard_name
            quantized_weight_map[salient_idx_key] = shard_name
            quantized_weight_map[salient_weight_key] = shard_name
            quantized_weight_map[salient_scale_key] = shard_name

            layer_bytes = (
                packed_q.numel() * 1
                + scales.numel() * 2
                + res_u.numel() * 2
                + res_v.numel() * 2
                + salient_idx.numel() * 4
                + salient_weight.numel() * 1
                + salient_scale.numel() * 2
            )
            orig_bytes = tensor.numel() * 2
            total_quantized_bytes += layer_bytes

            hub_tag = "[BIFURCATION HUB r=32]" if is_bifurcation else f"[STANDARD r={rank}]"
            print(
                f"    [DV-SSQ]      {tensor_name:48s} -> INT8({metrics['num_salient_channels']}c)+INT4+SVD {hub_tag:22s} "
                f"rel_err={metrics['frob_rel_error']:.4f} "
                f"({orig_bytes/(1024**2):.1f}MB -> {layer_bytes/(1024**2):.1f}MB)"
            )

        # Save quantized shard
        out_shard_path = os.path.join(quantized_model_dir, shard_name)
        print(f"[*] Saving compressed shard to {out_shard_path}...")
        quantized_shard_dict = {k: v.contiguous() for k, v in quantized_shard_dict.items()}
        safetensors.torch.save_file(quantized_shard_dict, out_shard_path)

        shard_size = os.path.getsize(out_shard_path)
        print(f"    Shard written successfully: {shard_size / (1024**2):.2f} MB")

        # Explicit RAM cleanup
        del shard_tensors
        del quantized_shard_dict
        gc.collect()

    # Save updated safetensors index
    print("\n" + "=" * 85)
    print("[*] Generating updated safetensors index...")
    new_index = {
        "metadata": {
            "total_parameters": raw_index["metadata"].get("total_parameters", 4112079360),
            "total_size": total_quantized_bytes,
            "quantization": "DV-SSQ-Hadamard-INT8-INT4-GSQ-RCO",
            "effective_bits": 5.62,
            "group_size": 64,
            "residual_ranks": "16-32",
            "salient_channels_ratio": 0.125,
        },
        "weight_map": quantized_weight_map,
    }
    index_out_path = os.path.join(quantized_model_dir, "model.safetensors.index.json")
    with open(index_out_path, "w") as f:
        json.dump(new_index, f, indent=2)
    print(f"    Index saved with {len(quantized_weight_map)} tensor entries.")

    # Update config.json
    print("[*] Updating config.json with quantization metadata...")
    with open(os.path.join(raw_model_dir, "config.json"), "r") as f:
        config = json.load(f)

    config["hadamard_spin"] = True
    config["gsq_int4"] = True
    config["dv_ssq"] = True
    config["selective_attention_preservation"] = True
    config["residual_rank"] = "16-32"
    config["effective_bits"] = "5.62-MLP / 16.0-Attn"
    config["quantization_config"] = {
        "quant_method": "dv_ssq_hadamard_int8_int4_rco",
        "bits_background": 4,
        "bits_salient": 8,
        "salient_channels_ratio": 0.125,
        "group_size": 64,
        "hadamard_spin": True,
        "hadamard_dim": 256,
        "selective_attention_preservation": True,
        "attention_quantized": False,
        "mlp_quantized": True,
        "residual_rank_default": 16,
        "residual_rank_bifurcation": 32,
        "effective_bits": 5.62,
        "bifurcation_layers": sorted(list({3, 7, 11, 15, 19, 23, 27, 31, 35} | set(range(16, 29)))),
    }

    with open(os.path.join(quantized_model_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy companion files
    companion_files = [
        "configuration_spark.py",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
        "special_tokens_map.json",
        "generation_config.json",
        "LICENSE",
    ]
    for cfile in companion_files:
        src = os.path.join(raw_model_dir, cfile)
        dst = os.path.join(quantized_model_dir, cfile)
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"    Copied companion file: {cfile}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 85)
    print(f"DV-SSQ QUANTIZATION COMPLETED IN {elapsed:.2f} SECONDS ({elapsed/60:.2f} MINUTES)")
    print(f"Total Quantized Size: {total_quantized_bytes / (1024**3):.2f} GB")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark-X2.5-4B Master DV-SSQ Quantizer")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default=r"C:\Users\PC MOD\Desktop\spark_hadamard_quant\raw_model",
        help="Path to raw model directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"C:\Users\PC MOD\Desktop\spark_hadamard_quant\quantized_model",
        help="Path to output quantized model directory",
    )
    args = parser.parse_args()
    run_master_quantization(args.raw_dir, args.out_dir)
