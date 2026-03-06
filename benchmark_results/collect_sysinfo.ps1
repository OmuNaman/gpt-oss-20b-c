$cpu = Get-CimInstance Win32_Processor
$os = Get-CimInstance Win32_OperatingSystem
$cs = Get-CimInstance Win32_ComputerSystem

$info = @"
=== GPT-OSS 20B Benchmark - System Info ===
Date: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
Computer: $env:COMPUTERNAME
OS: $($os.Caption) Build $($os.BuildNumber)
CPU: $($cpu.Name)
Cores (Physical): $($cpu.NumberOfCores)
Threads (Logical): $($cpu.NumberOfLogicalProcessors)
Max Clock: $($cpu.MaxClockSpeed) MHz
RAM: $([math]::Round($cs.TotalPhysicalMemory / 1GB, 1)) GB
"@

$info | Out-File -Encoding utf8 "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"

# Tool versions
"" | Add-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"
"=== Tool Versions ===" | Add-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"
$gccVer = gcc --version 2>&1 | Select-Object -First 1
"GCC: $gccVer" | Add-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"
$cmakeVer = cmake --version 2>&1 | Select-Object -First 1
"CMake: $cmakeVer" | Add-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"

try {
    $javaVer = java -version 2>&1 | Select-Object -First 1
    "Java: $javaVer" | Add-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"
} catch {
    "Java: NOT INSTALLED" | Add-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"
}

# CPU features via wmic
"" | Add-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"
"=== CPU Details ===" | Add-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"
$cpuDetails = Get-CimInstance Win32_Processor | Select-Object Name, Caption, Architecture, L2CacheSize, L3CacheSize
$cpuDetails | Format-List | Out-String | Add-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"

# Check available RAM right now
$freeRAM = [math]::Round(($os.FreePhysicalMemory / 1MB), 1)
"Available RAM (now): $freeRAM GB" | Add-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"

Write-Host "System info collected successfully."
Get-Content "D:\Coding_Workspace\GPT_OSS\benchmark_results\system_info.txt"
