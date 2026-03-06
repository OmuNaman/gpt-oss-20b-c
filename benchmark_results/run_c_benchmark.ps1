# C Engine Benchmark: 1 warmup + 5 measured runs
$modelPath = "D:\Coding_Workspace\GPT_OSS\raw_bin\gpt_oss_20b.bin"
$tokenizerPath = "D:\Coding_Workspace\GPT_OSS\raw_bin\tokenizer_gptoss.bin"
$exe = "D:\Coding_Workspace\GPT_OSS\run_gptoss_native.exe"
$outFile = "D:\Coding_Workspace\GPT_OSS\benchmark_results\c_engine.txt"

"=== run_gptoss.c (-march=native) ===" | Out-File -Encoding utf8 $outFile
"Date: $(Get-Date)" | Add-Content $outFile
"Prompt: Explain the halting problem in one paragraph." | Add-Content $outFile
"Max tokens: 256, Temp: 0, Seed: 42" | Add-Content $outFile
"" | Add-Content $outFile

$runs = @("warmup", "1", "2", "3", "4", "5")
foreach ($run in $runs) {
    Write-Host "--- C Engine Run: $run ---"
    "--- Run $run ---" | Add-Content $outFile
    $t = Measure-Command {
        $output = & $exe --model $modelPath --tokenizer $tokenizerPath --prompt "Explain the halting problem in one paragraph." --max_tokens 256 --temp 0 --seed 42 2>&1
    }
    $output | Out-String | Add-Content $outFile
    "Wall time: $($t.TotalSeconds) seconds" | Add-Content $outFile
    "" | Add-Content $outFile

    # Show progress
    $tokLine = $output | Select-String "Generation speed"
    Write-Host "  $tokLine  (wall: $([math]::Round($t.TotalSeconds, 1))s)"
}

Write-Host "`nC engine benchmark complete. Results in: $outFile"
