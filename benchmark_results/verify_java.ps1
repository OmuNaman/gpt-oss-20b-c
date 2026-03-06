# Refresh PATH from registry (picks up new JDK install)
$machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = "$machinePath;$userPath"

Write-Host "=== Java Version ==="
java -version 2>&1 | ForEach-Object { Write-Host $_ }

Write-Host "`n=== JAVA_HOME ==="
$jh = [System.Environment]::GetEnvironmentVariable("JAVA_HOME", "Machine")
Write-Host "JAVA_HOME: $jh"

Write-Host "`n=== Java Location ==="
where.exe java 2>&1 | ForEach-Object { Write-Host $_ }

# Append to system_info.txt
"Java: $(java -version 2>&1 | Select-Object -First 1)" | Add-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"
