param()
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Go to repo root (this script is expected to be in scripts/)
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..  # repo root

# Create/activate venv
if (-not (Test-Path ".\.venv")) { py -3.11 -m venv .venv }
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip wheel
pip install -r requirements.txt
pip install pyinstaller python-dotenv

# Detect chroma folder
$chromaParam = ""
if (Test-Path .\chroma_db) { $chromaParam = '--add-data "chroma_db;chroma_db"' }
elseif (Test-Path .\chroma) { $chromaParam = '--add-data "chroma;chroma"' }

# Build
$cmd = @"
pyinstaller --noconfirm --clean --onedir --console `
  --name CalRAG `
  --collect-all chromadb `
  --collect-all sentence_transformers `
  --collect-all tiktoken `
  --collect-all numpy `
  $chromaParam `
  launch_app.py
"@
powershell -NoProfile -Command $cmd

Write-Host "`nBuild complete. Output: dist\CalRAG\CalRAG.exe" -ForegroundColor Green


