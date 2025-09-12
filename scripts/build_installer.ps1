param()
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Go to repo root
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..

# Verify PyInstaller output exists
if (-not (Test-Path ".\dist\CalRAG\CalRAG.exe")) {
  throw "dist\CalRAG\CalRAG.exe not found. Run scripts\build_exe.ps1 first."
}

# Prefer existing installer script
$iss = "installer\CalRAGInstaller.iss"
if (-not (Test-Path $iss)) {
  New-Item -ItemType Directory -Force -Path "installer" | Out-Null
  $fallback = @"
#define MyAppName "Cal RAG Agent"
#define MyAppVersion "0.2.0"
#define MyAppExeName "CalRAG.exe"

[Setup]
AppId={{D5E8E984-5E6D-48E6-9F02-2B6A0C7F9A11}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={pf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputBaseFilename=CalRAGInstaller
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "dist\CalRAG\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
"@
  Set-Content -Path "installer\CalRAGInstaller.iss" -Value $fallback
}

# Compile with Inno Setup
$iscc = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) { $iscc = "${env:ProgramFiles}\Inno Setup 6\ISCC.exe" }
if (-not (Test-Path $iscc)) { throw "ISCC.exe not found. Install Inno Setup 6 and retry." }

& "$iscc" "installer\CalRAGInstaller.iss"
Write-Host "`nInstaller built. See installer\Output\CalRAGInstaller.exe" -ForegroundColor Green


