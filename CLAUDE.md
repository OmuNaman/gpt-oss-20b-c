# GPT-OSS 20B — C Inference Engine

## Project Overview

From-scratch single-file C inference engine for OpenAI's GPT-OSS 20B (MoE, 20.9B params, 3.6B active/token). Built in the llama2.c / nano_hindi tradition.

## Working Directory & OS

`D:\Coding_Workspace\GPT_OSS` — Windows 11, PowerShell, MinGW/gcc

## Architecture

- **Model**: GPT-OSS 20B — MoE, 32 experts, top-4 active per token
- **Weights**: MXFP4 (MoE experts, 91.4% of params) + float16 (attention)
- **Dims**: hidden=2880, layers=24, heads=64Q/8KV (GQA 8:1), head_dim=64
- **Vocab**: 201,088 (o200k_harmony, tiktoken-based BPE)
- **Context**: 4096 inference limit, native 131K via YaRN
- **Checkpoint**: 12.82 GB custom .bin (mmap'd)
- **Runtime**: ~386 MB (384 MB KV cache + ~2 MB activations)

## Files

```
run_gptoss.h                    — structs, constants, FP4_LUT, declarations
run_gptoss.c                    — full engine (~2000 lines)
run_gptoss_native.exe           — compiled binary (-march=native optimized)
raw_bin/gpt_oss_20b.bin         — model weights (12.82 GB)
raw_bin/tokenizer_gptoss.bin    — tokenizer binary
export_model.py                 — safetensors → .bin converter
export_tokenizer.py             — tiktoken → binary exporter
docs/POA.md                     — implementation plan
docs/GPT_OSS_DEEP_DIVE.md       — full model research
docs/devlog_1_simd_optimization.md — SIMD optimization plan
```

## Build (Windows MinGW)

```bash
# BEST — auto-detects CPU, enables all supported ISA (AVX2, FMA, F16C, BMI2, etc.)
# +18% faster than explicit -mavx2 -mfma -mf16c due to Raptor Lake-specific scheduling
gcc -O3 -march=native -fopenmp -o run_gptoss_native.exe run_gptoss.c -lm -lshell32

# Explicit SIMD (portable to any Haswell+ CPU, slightly slower than -march=native):
gcc -O3 -fopenmp -mavx2 -mfma -mf16c -o run_gptoss.exe run_gptoss.c -lm -lshell32

# Scalar fallback (any x86-64):
gcc -O3 -fopenmp -o run_gptoss_scalar.exe run_gptoss.c -lm -lshell32
```

**Why -march=native wins**: On Raptor Lake (i7-14700HX), it enables microarchitecture-specific instruction scheduling, BMI2, POPCNT, and lets the compiler make better register allocation decisions beyond just enabling AVX2/FMA/F16C.

## Run

```powershell
# Single prompt benchmark (deterministic)
./run_gptoss_native.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin --prompt "Explain the halting problem in one paragraph." --max_tokens 256 --temp 0 --seed 42

# Chat mode with reasoning
./run_gptoss_native.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin --chat --reasoning high --show-thinking

# Chat with specific params
./run_gptoss_native.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin --chat --reasoning high --show-thinking --temp 0.7 --max_tokens 512 --seed 42
```

## Measured Performance (REAL — March 2026)

**Hardware**: i7-14700HX (8P+12E cores, 28 threads), 16 GB DDR5, Windows 11, NVMe SSD
**Build**: `gcc -O3 -march=native -fopenmp` (Raptor Lake auto-detected)
**Test**: Chat, --reasoning high --show-thinking --temp 0.7 --max_tokens 512 --seed 42
**Prompt**: "Who are you?" (19 prompt tokens)

| Metric | Value |
|--------|-------|
| Generation speed | **3.92 tok/s** |
| Time per token | **255 ms** |
| Prompt speed | 3.1 tok/s (19 tokens in 6.07s) |
| Generated tokens | 86 (63 thinking + 14 response + control) |
| Total time | 27.99s |
| Model (mmap'd) | 12.82 GB |
| KV cache | 384 MB |
| Activations | 1.9 MB |
| Scalar baseline | 1.66 tok/s (604 ms/tok) |
| **SIMD speedup** | **2.36×** |

## Competitors

1. **gpt-oss.java** (Amazon) — github.com/amzn/gpt-oss.java — Pure Java, JDK 23+, ~1000 LOC, Vector API SIMD, reads SafeTensors directly
2. **llama.cpp** (ggml-org) — github.com/ggml-org/llama.cpp — C++, >100K LOC, MXFP4 GGUF native, multi-backend
3. No other from-scratch CPU engines exist for GPT-OSS 20B

## MXFP4 Quick Reference

- 4-bit E2M1, 2 per byte (low nibble=even, high nibble=odd)
- LUT: [0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
- E8M0 scale per group of 32: actual_scale = 2^(uint8 - 127)

## Known Issue

- print_metrics() duplicates some output lines (cosmetic bug)

## Code Style

- Single .c + .h (no file splitting)
- Scalar reference + SIMD overlay via #ifdef __AVX2__ / __F16C__
- OpenMP for parallelism
- mmap (Windows: CreateFileMapping/MapViewOfFile)
- Zero dynamic allocation during generation