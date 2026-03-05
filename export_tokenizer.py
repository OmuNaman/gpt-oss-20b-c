"""
Export the GPT-OSS o200k_harmony tokenizer to binary format for C inference.

Usage (on cloud):
    python export_tokenizer.py --output tokenizer_gptoss.bin

Binary format (all little-endian):
    [int32]  vocab_size         (201088)
    [int32]  max_token_length   (max byte length of any token piece)
    [int32]  eos_token_id       (200002)
    [int32]  pad_token_id       (199999)

    For each token (0 to vocab_size-1):
        [float32]  score       (negative rank: lower rank = higher priority = higher score)
        [int32]    len         (byte length of the piece)
        [bytes]    piece[len]  (raw byte string of the token)

This format is compatible with a modified version of the llama2.c tokenizer approach.
The BPE merge algorithm should prefer pairs with the HIGHEST score (= lowest rank).

Dependencies:
    pip install tiktoken transformers
"""

import argparse
import os
import struct
import sys


def export_with_tiktoken(output_path: str):
    """Export using tiktoken for accurate ranks."""
    import tiktoken

    print("Loading o200k_base tokenizer via tiktoken...")

    # o200k_base is the base encoding; o200k_harmony adds special tokens
    enc = tiktoken.get_encoding("o200k_base")

    # The base vocab
    # tiktoken stores: { base64_encoded_bytes: rank }
    # We also need the special tokens
    vocab_size = 201088  # full vocab including special tokens
    base_vocab_size = enc.n_vocab  # base vocab (without extra special tokens)

    print(f"  Base vocab size: {base_vocab_size}")
    print(f"  Target vocab size: {vocab_size}")

    # Build token_id -> (byte_string, rank) mapping
    # For tiktoken, the rank IS the token_id for the base vocabulary
    pieces = {}
    max_token_length = 0

    # Get the mergeable ranks (base vocabulary)
    # tiktoken._mergeable_ranks gives us token_bytes -> rank
    mergeable_ranks = enc._mergeable_ranks
    for token_bytes, rank in mergeable_ranks.items():
        pieces[rank] = token_bytes
        max_token_length = max(max_token_length, len(token_bytes))

    # Get special tokens
    special_tokens = enc._special_tokens
    for token_str, token_id in special_tokens.items():
        token_bytes = token_str.encode("utf-8")
        pieces[token_id] = token_bytes
        max_token_length = max(max_token_length, len(token_bytes))

    # Also try to get the full set of special tokens from transformers
    try:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
        # Get any additional special tokens
        if hasattr(hf_tok, 'added_tokens_encoder'):
            for token_str, token_id in hf_tok.added_tokens_encoder.items():
                if token_id not in pieces:
                    token_bytes = token_str.encode("utf-8")
                    pieces[token_id] = token_bytes
                    max_token_length = max(max_token_length, len(token_bytes))
                    print(f"  Added special token [{token_id}]: {token_str}")
    except Exception as e:
        print(f"  Note: Could not load HF tokenizer for extra special tokens: {e}")

    print(f"  Total tokens mapped: {len(pieces)}")
    print(f"  Max token length: {max_token_length} bytes")

    # Fill any gaps with placeholder tokens
    for i in range(vocab_size):
        if i not in pieces:
            # Unknown token — use a placeholder
            pieces[i] = f"<|reserved_{i}|>".encode("utf-8")
            max_token_length = max(max_token_length, len(pieces[i]))

    # Write binary
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        # Header
        f.write(struct.pack("iiii", vocab_size, max_token_length, 200002, 199999))

        for i in range(vocab_size):
            piece_bytes = pieces[i]
            # Score: use negative token_id as score proxy
            # For base tokens, token_id == rank, and lower rank = merge first
            # We use vocab_size - rank so that lower rank gets HIGHER score
            # This way the BPE merge loop can pick the highest-scoring pair
            score = float(vocab_size - i)
            f.write(struct.pack("f", score))
            f.write(struct.pack("i", len(piece_bytes)))
            f.write(piece_bytes)

    file_size = os.path.getsize(output_path)
    print(f"\nDone! Output: {output_path}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")

    # Print sample tokens
    print("\nSample tokens:")
    for i in [0, 1, 2, 3, 100, 1000, 10000, 50000, 200000, 200001, 200002]:
        if i < vocab_size and i in pieces:
            raw = pieces[i]
            try:
                display = raw.decode("utf-8", errors="replace")
            except:
                display = repr(raw)
            print(f"  [{i}] len={len(raw)} piece='{display[:50]}'")


def export_with_transformers_only(output_path: str):
    """Fallback: export using only the HuggingFace transformers tokenizer."""
    from transformers import AutoTokenizer

    print("Loading tokenizer via transformers (fallback)...")
    tok = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")

    vocab_size = 201088
    vocab = tok.get_vocab()
    id_to_piece = {v: k for k, v in vocab.items()}

    max_token_length = 0
    pieces = {}

    for i in range(vocab_size):
        if i in id_to_piece:
            piece_str = id_to_piece[i]
            piece_bytes = piece_str.encode("utf-8")
        else:
            try:
                piece_str = tok.decode([i])
                piece_bytes = piece_str.encode("utf-8")
            except:
                piece_bytes = f"<|reserved_{i}|>".encode("utf-8")

        pieces[i] = piece_bytes
        max_token_length = max(max_token_length, len(piece_bytes))

    print(f"  Total tokens: {len(pieces)}")
    print(f"  Max token length: {max_token_length} bytes")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(struct.pack("iiii", vocab_size, max_token_length, 200002, 199999))

        for i in range(vocab_size):
            piece_bytes = pieces[i]
            score = float(vocab_size - i)
            f.write(struct.pack("f", score))
            f.write(struct.pack("i", len(piece_bytes)))
            f.write(piece_bytes)

    file_size = os.path.getsize(output_path)
    print(f"\nDone! Output: {output_path}")
    print(f"File size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")


def export_tokenizer(output_path: str):
    """Export tokenizer, preferring tiktoken if available."""
    try:
        import tiktoken
        export_with_tiktoken(output_path)
    except ImportError:
        print("tiktoken not found, falling back to transformers-only export")
        export_with_transformers_only(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export GPT-OSS o200k tokenizer to binary format for C inference"
    )
    parser.add_argument(
        "--output", type=str, default="tokenizer_gptoss.bin",
        help="Output binary file path"
    )
    args = parser.parse_args()
    export_tokenizer(args.output)
