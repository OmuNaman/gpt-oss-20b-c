---
name: benchmark-gptoss
description: Benchmark GPT-OSS 20B CPU inference engines (C, Java, llama.cpp) on the local machine. Use when asked to run benchmarks, compare performance, measure tok/s, or collect data for the research paper.
---

# GPT-OSS 20B CPU Inference Benchmarking Skill

## Standard Benchmark Parameters

- **Prompt**: `"Explain the halting problem in one paragraph."`
- **Max tokens**: 256
- **Temperature**: 0 (greedy, deterministic)
- **Seed**: 42
- **Runs**: 5 measured + 1 warm-up (discard first)
- **Metrics**: tok/s, ms/token, prompt tok/s, TTFT, peak memory, wall time

## Step 1: System Profiling

```powershell
mkdir -Force benchmark_results
@"
Date: $(Get-Date)
Computer: $env:COMPUTERNAME
OS: $(Get-CimInstance Win32_OperatingSystem | Select-Object -ExpandProperty Caption)
CPU: $(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name)
Cores: $(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty NumberOfCores)
Threads: $(Get-CimInstance Win32_Processor | Select-Object -ExpandProperty NumberOfLogicalProcessors)
RAM: $([math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 1)) GB
"@ | Out-File benchmark_results/system_info.txt
gcc --version 2>&1 | Select-Object -First 1 | Add-Content benchmark_results/system_info.txt
java -version 2>&1 | Select-Object -First 1 | Add-Content benchmark_results/system_info.txt
cmake --version 2>&1 | Select-Object -First 1 | Add-Content benchmark_results/system_info.txt
```

## Step 2: Build Engines

### C Engine
Already built as `run_gptoss_native.exe`. If rebuild needed:
```bash
gcc -O3 -march=native -fopenmp -o run_gptoss_native.exe run_gptoss.c -lm -lshell32
```
`-march=native` auto-detects Raptor Lake and enables all ISA extensions with CPU-specific scheduling. **+18% over explicit -mavx2 -mfma -mf16c**.

### Java Engine
```powershell
if (-not (Test-Path gpt-oss-java)) { git clone https://github.com/amzn/gpt-oss.java.git gpt-oss-java }
cd gpt-oss-java; .\gradlew.bat build shadowJar 2>&1 | Tee-Object ../benchmark_results/java_build.txt; cd ..
```
Needs SafeTensors — if missing: `huggingface-cli download openai/gpt-oss-20b --include "original/*" --local-dir gpt-oss-20b`

### llama.cpp
```powershell
if (-not (Test-Path llama.cpp)) { git clone https://github.com/ggml-org/llama.cpp.git }
cd llama.cpp
cmake -B build -DGGML_AVX2=ON -DGGML_FMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $env:NUMBER_OF_PROCESSORS
cd ..
```
Needs GGUF — if missing: `huggingface-cli download ggml-org/gpt-oss-20b-GGUF gpt-oss-20b-mxfp4.gguf --local-dir .`

## Step 3: Run Benchmarks

### C Engine
```powershell
"=== run_gptoss.c (-march=native) ===" | Out-File benchmark_results/c_engine.txt
foreach ($run in @("warmup","1","2","3","4","5")) {
    "--- Run $run ---" | Add-Content benchmark_results/c_engine.txt
    $t = Measure-Command {
        ./run_gptoss_native.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin --prompt "Explain the halting problem in one paragraph." --max_tokens 256 --temp 0 --seed 42 2>&1 | Tee-Object -Variable out
    }
    $out | Add-Content benchmark_results/c_engine.txt
    "Wall: $($t.TotalSeconds)s" | Add-Content benchmark_results/c_engine.txt
}
```

### Java Engine
```powershell
$jar = Get-ChildItem -Recurse -Filter "gpt-oss-java-*-all.jar" | Select-Object -First 1
$st = Get-ChildItem -Recurse -Filter "model.safetensors" -Path "gpt-oss-20b" -ErrorAction SilentlyContinue | Select-Object -First 1
if ($jar -and $st) {
    "=== gpt-oss.java ===" | Out-File benchmark_results/java_engine.txt
    foreach ($run in @("warmup","1","2","3","4","5")) {
        "--- Run $run ---" | Add-Content benchmark_results/java_engine.txt
        java --add-modules jdk.incubator.vector -jar $jar.FullName $st.FullName -m generate -p "Explain the halting problem in one paragraph." -n 256 -t 0 --debug 2>&1 | Add-Content benchmark_results/java_engine.txt
    }
}
```

### llama.cpp
```powershell
$gguf = Get-ChildItem -Recurse -Filter "gpt-oss-20b*mxfp4*.gguf" | Select-Object -First 1
if ($gguf) {
    "=== llama.cpp ===" | Out-File benchmark_results/llamacpp_engine.txt
    # Structured bench
    ./llama.cpp/build/bin/Release/llama-bench.exe -m $gguf.FullName -p 128,512 -n 128 -r 5 2>&1 | Add-Content benchmark_results/llamacpp_engine.txt
    # Generation bench
    foreach ($run in @("warmup","1","2","3","4","5")) {
        "--- Run $run ---" | Add-Content benchmark_results/llamacpp_engine.txt
        ./llama.cpp/build/bin/Release/llama-cli.exe -m $gguf.FullName -p "Explain the halting problem in one paragraph." -n 256 --temp 0 -s 42 --no-display-prompt 2>&1 | Add-Content benchmark_results/llamacpp_engine.txt
    }
}
```

### Thread Scaling (C engine)
```powershell
"=== Thread Scaling ===" | Out-File benchmark_results/thread_scaling.txt
foreach ($t in @(1, 4, 8, 16)) {
    $env:OMP_NUM_THREADS = $t
    "--- Threads: $t ---" | Add-Content benchmark_results/thread_scaling.txt
    ./run_gptoss_native.exe --model raw_bin/gpt_oss_20b.bin --tokenizer raw_bin/tokenizer_gptoss.bin --prompt "Hello" --max_tokens 64 --temp 0 --seed 42 2>&1 | Select-String "tok/s|ms/tok|speed|time" | Add-Content benchmark_results/thread_scaling.txt
}
$env:OMP_NUM_THREADS = $null
```

## Step 4: Summarize to benchmark_results/SUMMARY.md

Include: system specs, build configs, results table (Engine | Avg tok/s | Std dev | ms/token | Peak RSS), thread scaling, llama-bench pp/tg, analysis of bottleneck (compute vs memory-bandwidth).

## Troubleshooting

- **Java too old**: `winget install EclipseAdoptium.Temurin.23.JDK`
- **cmake missing**: `winget install Kitware.CMake`
- **gradlew fails**: Use `.\gradlew.bat` not `./gradlew` on Windows
- **Weights missing**: Print download commands and STOP
- **OOM/killed**: Close browsers. 16GB is tight for 12.82GB model + 400MB runtime.
- **Slow first run**: Expected — mmap cold start. Always warm up.