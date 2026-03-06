# Refresh PATH from registry
$machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = "$machinePath;$userPath"

# Try java
$javaResult = where.exe java 2>&1
Write-Host "where.exe java: $javaResult"

# Also check JAVA_HOME
$jh = [System.Environment]::GetEnvironmentVariable("JAVA_HOME", "Machine")
Write-Host "JAVA_HOME (Machine): $jh"
$jh2 = [System.Environment]::GetEnvironmentVariable("JAVA_HOME", "User")
Write-Host "JAVA_HOME (User): $jh2"

# Brute force search
Get-ChildItem "C:\Program Files" -Directory -ErrorAction SilentlyContinue | ForEach-Object {
    $javaExe = Join-Path $_.FullName "bin\java.exe"
    if (Test-Path $javaExe) {
        Write-Host "FOUND: $javaExe"
    }
}
Get-ChildItem "C:\Program Files" -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*jdk*" -or $_.Name -like "*java*" -or $_.Name -like "*eclipse*" -or $_.Name -like "*temurin*" -or $_.Name -like "*adoptium*" } | ForEach-Object {
    Write-Host "DIR: $($_.FullName)"
}

# Check winget
winget list --name "Temurin" 2>&1 | Write-Host
