"""
Export OpenAI GPT-OSS 20B from SafeTensors (original/ format) to flat binary.

Usage (on cloud GPU):
    python export_model.py --input gpt-oss-20b-raw/original --output gpt_oss_20b.bin

Binary format:
    Header: 16 x int32 (64 bytes)
    Weights: contiguous arrays in forward-pass order
        - BF16 tensors → converted to float16
        - MXFP4 blocks → kept as raw uint8
        - MXFP4 scales → kept as raw uint8

See GPT_OSS_DEEP_DIVE.md for complete format specification.
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

# Try torch first (for BF16 support), fall back to manual conversion
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: torch not found. Using manual BF16 conversion (slower).")

from safetensors import safe_open


# ─── Header constants ───────────────────────────────────────────────────────
MAGIC = 0x474F5353   # "GOSS" in ASCII
VERSION = 1


def bf16_bytes_to_f16(raw_bytes: bytes) -> np.ndarray:
    """Convert raw bfloat16 bytes to float16 numpy array.

    BF16 = sign(1) + exponent(8) + mantissa(7)
    FP16 = sign(1) + exponent(5) + mantissa(10)

    We go through float32 as intermediate to handle the exponent range difference.
    """
    bf16 = np.frombuffer(raw_bytes, dtype=np.uint16)
    # BF16 → float32: just shift left by 16 bits
    f32_bits = bf16.astype(np.uint32) << 16
    f32 = f32_bits.view(np.float32)
    # float32 → float16 (numpy handles overflow → inf correctly)
    return f32.astype(np.float16)


def load_tensor_as_f16(sf, name: str) -> np.ndarray:
    """Load a BF16 tensor from SafeTensors and convert to float16."""
    if HAS_TORCH:
        # Use torch for clean BF16 handling
        with safe_open(sf, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(name)
        if tensor.dtype == torch.bfloat16:
            return tensor.to(torch.float16).numpy()
        elif tensor.dtype == torch.float16:
            return tensor.numpy()
        elif tensor.dtype == torch.float32:
            return tensor.to(torch.float16).numpy()
        else:
            raise ValueError(f"Unexpected dtype {tensor.dtype} for {name}")
    else:
        # Manual: read raw bytes and convert
        with safe_open(sf, framework="numpy", device="cpu") as f:
            tensor = f.get_tensor(name)
        if tensor.dtype == np.float16:
            return tensor
        elif tensor.dtype == np.float32:
            return tensor.astype(np.float16)
        else:
            # Assume BF16 stored as uint16
            return bf16_bytes_to_f16(tensor.tobytes()).reshape(tensor.shape)


def load_tensor_raw(sf, name: str) -> np.ndarray:
    """Load a tensor as raw bytes (for MXFP4 blocks and UE8 scales)."""
    if HAS_TORCH:
        with safe_open(sf, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(name)
        return tensor.numpy().astype(np.uint8)
    else:
        with safe_open(sf, framework="numpy", device="cpu") as f:
            tensor = f.get_tensor(name)
        return tensor.astype(np.uint8)


def export_model(input_dir: str, output_path: str):
    """Export GPT-OSS 20B weights from original/ format to flat binary."""

    # ─── Load config ────────────────────────────────────────────────────────
    config_path = os.path.join(input_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        print(f"WARNING: {config_path} not found, using defaults for 20B")
        config = {}

    # Extract config values (with defaults for 20B)
    hidden_size = config.get("hidden_size", 2880)
    intermediate_size = config.get("intermediate_size", 2880)
    num_layers = config.get("num_hidden_layers", 24)
    num_heads = config.get("num_attention_heads", 64)
    num_kv_heads = config.get("num_key_value_heads", 8)
    head_dim = config.get("head_dim", 64)
    vocab_size = config.get("vocab_size", 201088)
    num_experts = config.get("num_experts", 32)
    experts_per_token = config.get("experts_per_token", 4)
    sliding_window = config.get("sliding_window", 128)
    max_seq_len = 4096  # inference limit (not the 131K training context)
    rope_theta = int(config.get("rope_theta", 150000))
    eos_token_id = 200002

    # Derived constants
    qkv_dim = head_dim * (num_heads + 2 * num_kv_heads)  # 5120
    attn_out_dim = head_dim * num_heads  # 4096
    mlp1_out_dim = intermediate_size * 2  # 5760
    mlp1_packed_cols = hidden_size // 2   # 1440
    num_groups_in = hidden_size // 32     # 90 (MXFP4 group size = 32)
    num_groups_mid = intermediate_size // 32  # 90

    print(f"\nModel config:")
    print(f"  hidden_size       = {hidden_size}")
    print(f"  intermediate_size = {intermediate_size}")
    print(f"  num_layers        = {num_layers}")
    print(f"  num_heads         = {num_heads}")
    print(f"  num_kv_heads      = {num_kv_heads}")
    print(f"  head_dim          = {head_dim}")
    print(f"  vocab_size        = {vocab_size}")
    print(f"  num_experts       = {num_experts}")
    print(f"  experts_per_token = {experts_per_token}")
    print(f"  sliding_window    = {sliding_window}")
    print(f"  qkv_dim           = {qkv_dim}")
    print(f"  attn_out_dim      = {attn_out_dim}")
    print(f"  mlp1_out_dim      = {mlp1_out_dim}")
    print(f"  mlp1_packed_cols  = {mlp1_packed_cols}")

    # ─── Find SafeTensors file ──────────────────────────────────────────────
    sf_files = [f for f in os.listdir(input_dir) if f.endswith(".safetensors")]
    if not sf_files:
        print(f"ERROR: No .safetensors files found in {input_dir}")
        sys.exit(1)

    sf_path = os.path.join(input_dir, sf_files[0])
    print(f"\nReading from: {sf_path}")
    print(f"File size: {os.path.getsize(sf_path) / 1024**3:.2f} GB")

    # ─── List all tensors ───────────────────────────────────────────────────
    if HAS_TORCH:
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            all_keys = list(f.keys())
    else:
        with safe_open(sf_path, framework="numpy", device="cpu") as f:
            all_keys = list(f.keys())

    print(f"\nFound {len(all_keys)} tensors:")
    for k in sorted(all_keys)[:20]:
        print(f"  {k}")
    if len(all_keys) > 20:
        print(f"  ... and {len(all_keys) - 20} more")

    # ─── Write binary ──────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total_bytes = 0
    start_time = time.time()

    with open(output_path, "wb") as out:

        # ── Header (64 bytes = 16 x int32) ──────────────────────────────────
        header = struct.pack("16i",
            MAGIC,
            VERSION,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            vocab_size,
            num_experts,
            experts_per_token,
            sliding_window,
            max_seq_len,
            rope_theta,
            eos_token_id,
            0,  # reserved
        )
        out.write(header)
        total_bytes += len(header)
        print(f"\n  Header: {len(header)} bytes")

        # ── Section 1: Embedding ─────────────────────────────────────────────
        print(f"\n  Embedding...")
        w = load_tensor_as_f16(sf_path, "embedding.weight")
        assert w.shape == (vocab_size, hidden_size), f"embedding shape {w.shape}"
        out.write(w.tobytes())
        total_bytes += w.nbytes
        print(f"    embedding.weight: {w.shape} float16 = {w.nbytes:,} bytes")

        # ── Section 2: Transformer Layers ────────────────────────────────────
        for layer in range(num_layers):
            layer_start = total_bytes
            prefix = f"block.{layer}"
            print(f"\n  Layer {layer}/{num_layers}...")

            # --- Attention ---
            # attn.norm.scale
            w = load_tensor_as_f16(sf_path, f"{prefix}.attn.norm.scale")
            assert w.shape == (hidden_size,), f"attn norm shape {w.shape}"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # attn.qkv.weight
            w = load_tensor_as_f16(sf_path, f"{prefix}.attn.qkv.weight")
            assert w.shape == (qkv_dim, hidden_size), f"qkv weight shape {w.shape}"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # attn.qkv.bias
            w = load_tensor_as_f16(sf_path, f"{prefix}.attn.qkv.bias")
            assert w.shape == (qkv_dim,), f"qkv bias shape {w.shape}"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # attn.sinks
            w = load_tensor_as_f16(sf_path, f"{prefix}.attn.sinks")
            assert w.shape == (num_heads,), f"sinks shape {w.shape}"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # attn.out.weight
            w = load_tensor_as_f16(sf_path, f"{prefix}.attn.out.weight")
            assert w.shape == (hidden_size, attn_out_dim), f"out weight shape {w.shape}"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # attn.out.bias
            w = load_tensor_as_f16(sf_path, f"{prefix}.attn.out.bias")
            assert w.shape == (hidden_size,), f"out bias shape {w.shape}"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            print(f"    Attention: {(total_bytes - layer_start):,} bytes")
            attn_end = total_bytes

            # --- MoE ---
            # mlp.norm.scale
            w = load_tensor_as_f16(sf_path, f"{prefix}.mlp.norm.scale")
            assert w.shape == (hidden_size,), f"mlp norm shape {w.shape}"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # mlp.gate.weight
            w = load_tensor_as_f16(sf_path, f"{prefix}.mlp.gate.weight")
            assert w.shape == (num_experts, hidden_size), f"gate weight shape {w.shape}"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # mlp.gate.bias
            w = load_tensor_as_f16(sf_path, f"{prefix}.mlp.gate.bias")
            assert w.shape == (num_experts,), f"gate bias shape {w.shape}"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # mlp1 blocks (MXFP4 — raw uint8, stored as 4D: experts × rows × groups × 16)
            w = load_tensor_raw(sf_path, f"{prefix}.mlp.mlp1_weight.blocks")
            assert w.shape == (num_experts, mlp1_out_dim, num_groups_in, 16), \
                f"mlp1 blocks shape {w.shape} != ({num_experts}, {mlp1_out_dim}, {num_groups_in}, 16)"
            w = w.reshape(num_experts, mlp1_out_dim, -1)  # flatten to (32, 5760, 1440)
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # mlp1 scales (E8M0 — raw uint8, one per group: experts × rows × groups)
            w = load_tensor_raw(sf_path, f"{prefix}.mlp.mlp1_weight.scales")
            assert w.shape == (num_experts, mlp1_out_dim, num_groups_in), \
                f"mlp1 scales shape {w.shape} != ({num_experts}, {mlp1_out_dim}, {num_groups_in})"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # mlp1 bias (BF16 → float16)
            w = load_tensor_as_f16(sf_path, f"{prefix}.mlp.mlp1_bias")
            assert w.shape == (num_experts, mlp1_out_dim), \
                f"mlp1 bias shape {w.shape} != ({num_experts}, {mlp1_out_dim})"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # mlp2 blocks (MXFP4 — raw uint8, stored as 4D: experts × rows × groups × 16)
            w = load_tensor_raw(sf_path, f"{prefix}.mlp.mlp2_weight.blocks")
            assert w.shape == (num_experts, hidden_size, num_groups_mid, 16), \
                f"mlp2 blocks shape {w.shape} != ({num_experts}, {hidden_size}, {num_groups_mid}, 16)"
            w = w.reshape(num_experts, hidden_size, -1)  # flatten to (32, 2880, 1440)
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # mlp2 scales (E8M0 — raw uint8, one per group: experts × rows × groups)
            w = load_tensor_raw(sf_path, f"{prefix}.mlp.mlp2_weight.scales")
            assert w.shape == (num_experts, hidden_size, num_groups_mid), \
                f"mlp2 scales shape {w.shape} != ({num_experts}, {hidden_size}, {num_groups_mid})"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            # mlp2 bias (BF16 → float16)
            w = load_tensor_as_f16(sf_path, f"{prefix}.mlp.mlp2_bias")
            assert w.shape == (num_experts, hidden_size), \
                f"mlp2 bias shape {w.shape} != ({num_experts}, {hidden_size})"
            out.write(w.tobytes())
            total_bytes += w.nbytes

            layer_bytes = total_bytes - layer_start
            moe_bytes = total_bytes - attn_end
            print(f"    MoE:       {moe_bytes:,} bytes")
            print(f"    Layer total: {layer_bytes:,} bytes ({layer_bytes / 1024**2:.1f} MB)")

        # ── Section 3: Final Norm + Unembedding ─────────────────────────────
        print(f"\n  Final norm + unembedding...")

        # norm.scale
        w = load_tensor_as_f16(sf_path, "norm.scale")
        assert w.shape == (hidden_size,), f"final norm shape {w.shape}"
        out.write(w.tobytes())
        total_bytes += w.nbytes
        print(f"    norm.scale: {w.shape}")

        # unembedding.weight
        w = load_tensor_as_f16(sf_path, "unembedding.weight")
        assert w.shape == (vocab_size, hidden_size), f"unembed shape {w.shape}"
        out.write(w.tobytes())
        total_bytes += w.nbytes
        print(f"    unembedding.weight: {w.shape} = {w.nbytes:,} bytes")

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path)
    assert file_size == total_bytes, f"Size mismatch: file={file_size}, written={total_bytes}"

    print(f"\n{'='*60}")
    print(f"Export complete!")
    print(f"  Output:    {output_path}")
    print(f"  File size: {file_size:,} bytes ({file_size / 1024**3:.2f} GB)")
    print(f"  Time:      {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export GPT-OSS 20B to flat binary for C inference engine"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Path to the original/ directory (containing model.safetensors)"
    )
    parser.add_argument(
        "--output", type=str, default="gpt_oss_20b.bin",
        help="Output binary file path"
    )
    args = parser.parse_args()
    export_model(args.input, args.output)
