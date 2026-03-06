# Try installing JDK 23 with elevated privileges
Write-Host "=== Attempting JDK 23 Install ==="
try {
    $result = winget list --name "Temurin" 2>&1
    if ($result -match "23") {
        Write-Host "Temurin 23 already in winget list"
    } else {
        Write-Host "Installing Temurin JDK 23..."
        winget install EclipseAdoptium.Temurin.23.JDK --accept-source-agreements --accept-package-agreements --force 2>&1
    }
} catch {
    Write-Host "winget failed: $_"
}

# Refresh PATH from system
$machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = "$machinePath;$userPath"

# Try to find java
Write-Host "`n=== Java Check ==="
$javaExe = where.exe java 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Java found: $javaExe"
    java -version 2>&1
} else {
    Write-Host "Java still not on PATH after refresh"
    # Manual search
    $found = Get-ChildItem "C:\Program Files" -Recurse -Filter "java.exe" -Depth 4 -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($found) {
        Write-Host "Java found at: $($found.FullName)"
        & $found.FullName -version 2>&1
    } else {
        Write-Host "Java NOT FOUND anywhere in Program Files"
    }
}

# HuggingFace CLI check
Write-Host "`n=== HuggingFace CLI Check ==="
$hfCli = python -c "import shutil; print(shutil.which('huggingface-cli') or 'NOT_FOUND')" 2>&1
Write-Host "huggingface-cli: $hfCli"

# Try via python -m
python -m huggingface_hub version 2>&1
