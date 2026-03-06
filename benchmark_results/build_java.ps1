# Refresh PATH to include JDK 23
$machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = "$machinePath;$userPath"
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-23.0.2.7-hotspot"

Write-Host "Java version:"
java -version 2>&1

Write-Host "`nBuilding gpt-oss.java..."
Set-Location "D:\Coding_Workspace\GPT_OSS\gpt-oss-java"
.\gradlew.bat build shadowJar 2>&1

Write-Host "`nLooking for JAR..."
Get-ChildItem -Recurse -Filter "*-all.jar" | Select-Object -ExpandProperty FullName
