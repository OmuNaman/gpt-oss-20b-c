# Java Engine Benchmark: 1 warmup + 5 measured runs
$machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = "$machinePath;$userPath"
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-23.0.2.7-hotspot"

$jar = "D:\Coding_Workspace\GPT_OSS\gpt-oss-java\build\libs\gpt-oss-java-1.0.0-all.jar"
$model = "D:\Coding_Workspace\GPT_OSS\gpt-oss-20b-safetensors\original\model.safetensors"
$outFile = "D:\Coding_Workspace\GPT_OSS\benchmark_results\java_engine.txt"

"=== gpt-oss.java Benchmark ===" | Out-File -Encoding utf8 $outFile
"Date: $(Get-Date)" | Add-Content $outFile
"Prompt: Explain the halting problem in one paragraph." | Add-Content $outFile
"Max tokens: 256, Temp: 0, Heap: -Xmx14g" | Add-Content $outFile
"" | Add-Content $outFile

$runs = @("warmup", "1", "2", "3", "4", "5")
foreach ($run in $runs) {
    Write-Host "--- Java Engine Run: $run ---"
    "--- Run $run ---" | Add-Content $outFile
    $t = Measure-Command {
        $output = & java -Xmx14g --add-modules jdk.incubator.vector -jar $jar $model -m generate -p "Explain the halting problem in one paragraph." -n 256 -t 0 --debug 2>&1
    }
    $output | Out-String | Add-Content $outFile
    "Wall time: $($t.TotalSeconds) seconds" | Add-Content $outFile
    "" | Add-Content $outFile

    # Show progress
    $tokLine = $output | Select-String "tok/s|token/s|speed"
    Write-Host "  $tokLine  (wall: $([math]::Round($t.TotalSeconds, 1))s)"
}

Write-Host "`nJava engine benchmark complete. Results in: $outFile"
