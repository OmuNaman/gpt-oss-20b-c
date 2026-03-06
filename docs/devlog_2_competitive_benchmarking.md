# Devlog 2: Competitive CPU Benchmarking

> **Goal**: Benchmark our from-scratch C inference engine against every known CPU
> competitor for GPT-OSS 20B — Amazon's gpt-oss.java and ggml-org's llama.cpp —
> on identical hardware with identical prompts to produce fair, reproducible numbers
> suitable for a research paper.

> **Hardware**: Intel Core i7-14700HX (8P + 12E, 28 threads), 16 GB DDR5, NVMe SSD, Windows 11
>
> **Date**: March 6, 2026

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [The Competitors](#2-the-competitors)
3. [Setup & Build Process](#3-setup--build-process)
4. [Benchmark Protocol](#4-benchmark-protocol)
5. [Results](#5-results)
6. [Analysis](#6-analysis)

---

## 1. Motivation

After Devlog 1, our C engine hit **3.92 tok/s** (warm cache) and **2.36 tok/s** (cold) —
a 2.36x improvement over the 1.66 tok/s scalar baseline. But the question remained:
**how does a 2000-line from-scratch C engine compare to production frameworks?**

The two known CPU competitors for GPT-OSS 20B are:

| Engine | Language | LOC | Repo |
|--------|----------|-----|------|
| **run_gptoss.c** (ours) | C | ~2000 | [OmuNaman/gpt-oss-20b-c](https://github.com/OmuNaman/gpt-oss-20b-c) |
| **gpt-oss.java** (Amazon) | Java | ~1000 | [amzn/gpt-oss.java](https://github.com/amzn/gpt-oss.java) |
| **llama.cpp** (ggml-org) | C++ | >100K | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |

We also searched GitHub for other from-scratch implementations in Rust, Go, Zig, etc.
Found [rusty-gpt](https://github.com/chaoz2/rusty-gpt) (Rust HTTP server wrapper) and
[HamzaElshafie/gpt-oss-20B](https://github.com/HamzaElshafie/gpt-oss-20B) (PyTorch
educational), but neither is a viable CPU inference benchmark target. **These three
are the only CPU engines that exist for GPT-OSS 20B.**

---

## 2. The Competitors

### 2a. Our C Engine (`run_gptoss.c`)

- **What it is**: Single-file (~2000 LOC) pure C inference engine
- **Quantization**: Custom MXFP4 binary format (12.82 GB), mmap'd
- **SIMD**: Hand-written AVX2/FMA/F16C intrinsics (see Devlog 1)
- **Parallelism**: OpenMP (sequential experts, inner-parallel matmuls)
- **Build**: `gcc -O3 -march=native -fopenmp` (GCC 15.2, MinGW)
- **Binary size**: 112 KB

### 2b. Amazon gpt-oss.java

- **What it is**: Pure Java inference engine (~1000 LOC) by Amazon
- **Quantization**: Reads SafeTensors directly (MXFP4 native format, 13 GB)
- **SIMD**: Java Vector API (`jdk.incubator.vector`) — JVM auto-vectorization
- **Parallelism**: Java threads (ForkJoinPool)
- **Build**: Gradle shadowJar, JDK 23+ required (Temurin 23.0.2)
- **JAR size**: ~30 MB (fat JAR with all dependencies)

### 2c. llama.cpp

- **What it is**: Production C++ inference framework (>100K LOC)
- **Quantization**: GGUF format with MXFP4 support (11.28 GB)
- **SIMD**: Auto-detected via cmake (`-march=native`)
- **Parallelism**: Custom thread pool, OpenMP for BLAS
- **Build**: cmake + ninja, Release mode
- **Binary size**: ~11 MB (llama-completion), ~5.5 MB (llama-bench)

---

## 3. Setup & Build Process

### 3a. System Preparation

```
OS:      Windows 11 Home Build 26200
CPU:     Intel Core i7-14700HX (8P+12E, 20 cores, 28 threads)
RAM:     15.7 GB DDR5
GCC:     15.2.0 (MinGW-W64 x86_64-ucrt-posix-seh)
Java:    OpenJDK 23.0.2 (Temurin-23.0.2+7)
CMake:   4.2.1
Disk:    163 GB free on NVMe SSD
```

### 3b. Installing Dependencies

1. **JDK 23**: `winget install EclipseAdoptium.Temurin.23.JDK` — needed for gpt-oss.java.
   The Java Vector API (`jdk.incubator.vector`) requires JDK 23+.

2. **huggingface-hub**: `pip install huggingface-hub` — for downloading model weights.
   Had to uninstall `hf_xet` (HuggingFace's new Xet storage backend) because it
   crashed with `RuntimeError: cannot convert value 150 to CompressionScheme`.
   Standard HTTP download worked fine after removing it.

### 3c. Downloading Model Weights

Three different formats for the same model:

| Format | File | Size | Engine |
|--------|------|------|--------|
| Custom .bin (MXFP4) | `raw_bin/gpt_oss_20b.bin` | 12.82 GB | C engine |
| SafeTensors (MXFP4) | `gpt-oss-20b-safetensors/original/model.safetensors` | 13.0 GB | Java |
| GGUF (MXFP4) | `gpt-oss-20b-gguf/gpt-oss-20b-mxfp4.gguf` | 11.28 GB | llama.cpp |

All three contain the same model weights in MXFP4 quantization — just different
container formats. Size differences come from metadata overhead and packing.

Downloaded from HuggingFace:
- SafeTensors: `openai/gpt-oss-20b` (original/ folder) — 3.68 GB/s download
- GGUF: `ggml-org/gpt-oss-20b-GGUF` (gpt-oss-20b-mxfp4.gguf)

### 3d. Building Competitors

**gpt-oss.java**:
```powershell
git clone --depth 1 https://github.com/amzn/gpt-oss.java.git gpt-oss-java
cd gpt-oss-java
.\gradlew.bat build shadowJar   # BUILD SUCCESSFUL in 50m 28s
# Output: build/libs/gpt-oss-java-1.0.0-all.jar
```

**llama.cpp**:
```powershell
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_AVX2=ON -DGGML_FMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target llama-bench llama-completion -j 20
```

**Build issue**: llama.cpp's vendored `cpp-httplib` uses `CreateFile2` (Windows 8+ API)
which MinGW's headers don't declare. Patched `httplib.cpp` to use `CreateFileW` +
`CreateFileMappingW` instead. Only affects the HTTP server, not inference.

---

## 4. Benchmark Protocol

### Standard Parameters (identical for all engines)

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Prompt** | "Explain the halting problem in one paragraph." | Non-trivial, forces reasoning |
| **Max tokens** | 256 | Long enough to measure sustained throughput |
| **Temperature** | 0 | Greedy/deterministic — eliminates sampling variance |
| **Seed** | 42 | Reproducibility |
| **Warm-up** | 1 run (discarded) | Loads model into OS page cache |
| **Measured runs** | 5 | Statistical significance |

### Metrics Collected

- **Generation tok/s**: Tokens per second during generation (main metric)
- **ms/token**: Milliseconds per generated token
- **Prompt tok/s**: Prompt processing speed (prefill)
- **TTFT**: Time to first token
- **Wall time**: Total end-to-end time
- **Peak RSS**: Memory usage

### Engine-Specific Commands

**C Engine:**
```powershell
./run_gptoss_native.exe --model raw_bin/gpt_oss_20b.bin `
  --tokenizer raw_bin/tokenizer_gptoss.bin `
  --prompt "Explain the halting problem in one paragraph." `
  --max_tokens 256 --temp 0 --seed 42
```

**Java Engine:**
```powershell
java --add-modules jdk.incubator.vector `
  -jar gpt-oss-java/build/libs/gpt-oss-java-1.0.0-all.jar `
  gpt-oss-20b-safetensors/original/model.safetensors `
  -m generate -p "Explain the halting problem in one paragraph." `
  -n 256 -t 0 --debug
```

**llama.cpp:**
```powershell
./llama.cpp/build/bin/llama-completion.exe `
  -m gpt-oss-20b-gguf/gpt-oss-20b-mxfp4.gguf `
  -p "Explain the halting problem in one paragraph." `
  -n 256 --temp 0 -s 42 --no-display-prompt
```

### Additional Tests

**Thread scaling (C engine only):**
```powershell
# OMP_NUM_THREADS = 1, 4, 8, 16
# --max_tokens 64, same prompt
```

**llama-bench (structured benchmark):**
```powershell
./llama.cpp/build/bin/llama-bench.exe `
  -m gpt-oss-20b-gguf/gpt-oss-20b-mxfp4.gguf `
  -p 128,512 -n 128 -r 5
```

### Fairness Notes

- All engines run on the **same machine** at the **same time** (sequentially, not simultaneously)
- All use the **same MXFP4 quantization** — just different container formats
- All have access to **AVX2/FMA/F16C** (same ISA)
- All use **mmap** or equivalent for model loading
- The 16 GB RAM constraint affects all engines equally (12-13 GB model + ~400 MB runtime)
- First run (warm-up) loads model into page cache; measured runs benefit equally

---

## 5. Results

> **Status: PENDING** — Benchmarks not yet run. Results will be added here and in
> `benchmark_results/SUMMARY.md` after execution.

### 5a. C Engine Results

| Run | Generation tok/s | ms/token | Prompt tok/s | Wall time |
|-----|-----------------|----------|-------------|-----------|
| warmup | — | — | — | — |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| **Average** | | | | |
| **Std dev** | | | | |

### 5b. Java Engine Results

| Run | Generation tok/s | ms/token | Prompt tok/s | Wall time |
|-----|-----------------|----------|-------------|-----------|
| warmup | — | — | — | — |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| **Average** | | | | |
| **Std dev** | | | | |

### 5c. llama.cpp Results

| Run | Generation tok/s | ms/token | Prompt tok/s | Wall time |
|-----|-----------------|----------|-------------|-----------|
| warmup | — | — | — | — |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |
| 5 | | | | |
| **Average** | | | | |
| **Std dev** | | | | |

### 5d. Thread Scaling (C Engine)

| Threads | Generation tok/s | ms/token | vs Default |
|---------|-----------------|----------|-----------|
| 1 | | | |
| 4 | | | |
| 8 | | | |
| 16 | | | |

### 5e. llama-bench (Structured)

| Test | pp (tok/s) | tg (tok/s) |
|------|-----------|-----------|
| pp128 | | |
| pp512 | | |
| tg128 | | |

---

## 6. Analysis

> **Status: PENDING** — Will cover:
> - Which engine wins and by how much
> - Compute-bound vs memory-bound assessment
> - LOC efficiency (tok/s per 1000 lines of code)
> - How 2000-line C compares to 100K+ line C++
> - Research paper readiness
> - Suggested follow-up experiments

---

## Appendix: File Locations

```
benchmark_results/
  system_info.txt         — Hardware and software versions
  c_engine.txt            — Raw C engine output (6 runs)
  java_engine.txt         — Raw Java engine output (6 runs)
  llamacpp_engine.txt     — Raw llama.cpp output + llama-bench
  thread_scaling.txt      — C engine thread sweep
  SUMMARY.md              — Final comparison table + analysis
```

## Appendix: Build Artifacts

```
run_gptoss_native.exe                              — C engine (112 KB)
gpt-oss-java/build/libs/gpt-oss-java-1.0.0-all.jar — Java engine (~30 MB)
llama.cpp/build/bin/llama-bench.exe                 — llama.cpp bench (5.5 MB)
llama.cpp/build/bin/llama-completion.exe             — llama.cpp inference (11 MB)
```
