## Cal RAG Agent – Windows Build & Installer Guide

This guide explains how to build a single-file Windows app with PyInstaller and package it into a Windows installer (Inno Setup). The installer ships your pre-ingested ChromaDB as initial data and creates Start Menu/Desktop shortcuts.

### 1) Prerequisites on the build machine

- Windows 10/11 (recommended for building Windows binaries)
- Python 3.11+ matching your project
- Node.js installed (Playwright requires it at runtime for browser downloads)
- Inno Setup 6+ (`iscc.exe` on PATH)

### 2) One-time Project Prep

In the repo root:

```powershell
pip install -r requirements.txt
pip install pyinstaller
```

Optional: place Tesseract under `vendor/tesseract/tesseract.exe` if you want to ship it. Otherwise, the app expects a system install or you can set `TESSERACT_EXE` later.

### 3) Build the app (PyInstaller)

The spec builds `windows_launcher.py` into `dist/CalRAG/CalRAG.exe`.

```powershell
pyinstaller --clean --noconfirm CalRAG.spec
```

Artifacts:
- `dist/CalRAG/CalRAG.exe` – the app entry point (launches Streamlit UI)
- `dist/CalRAG/` – bundled libs and your modules

### 4) Prepare bundled data

Ensure your pre-ingested ChromaDB folder exists at `chroma_db/` in the repo root. The installer will copy this to `%LOCALAPPDATA%\CalRAG\chroma` on the target machine.

Rules JSON under `rules/` are included automatically by the spec.

### 5) Build the installer (Inno Setup)

Create `installer/CalRAGInstaller.iss` (see example content below). Then run:

```powershell
iscc installer\CalRAGInstaller.iss
```

Produces: `installer\Output\CalRAGInstaller.exe` (path may vary depending on your Inno settings). Ship this file.

### 6) First run behavior (target machine)

- Installer copies the app to `C:\Program Files\Cal RAG Agent\`.
- Preloaded Chroma data goes to `%LOCALAPPDATA%\CalRAG\chroma`.
- First launch creates `%LOCALAPPDATA%\CalRAG\.env` if missing, with defaults:
  - `MODEL_CHOICE=gpt-4.1-mini`
  - `OPENAI_API_KEY=` (you fill this in if using OpenAI)
- The app opens the default browser at `http://localhost:8501`.

### 7) Playwright and Tesseract

- Playwright: During install, a hidden postinstall run executes `CalRAG.exe --install-playwright` to download the Chromium browser once. If it fails (e.g., offline), ingestion features needing a browser will wait until you run it again with internet.
- Tesseract: If you ship binaries under `{app}\tesseract\tesseract.exe`, the launcher sets `TESSERACT_EXE` automatically. Otherwise, install Tesseract system-wide or set `TESSERACT_EXE` in `%LOCALAPPDATA%\CalRAG\.env`.

### 8) Updating builds

- Rebuild PyInstaller and re-run Inno Setup. User data in `%LOCALAPPDATA%\CalRAG` persists across updates.

---

### Example Inno Setup script: `installer/CalRAGInstaller.iss`

Adjust paths if your repo layout differs.

```ini
; Inno Setup Script for Cal RAG Agent

#define MyAppName "Cal RAG Agent"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "Your Company"
#define MyAppExeName "CalRAG.exe"

[Setup]
AppId={{B9ADF0B5-9B18-4E94-98D3-3F3E6F3C3F01}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={pf}\Cal RAG Agent
DisableDirPage=no
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=no
OutputDir=installer\Output
OutputBaseFilename=CalRAGInstaller
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; PyInstaller one-dir output to Program Files
Source: "dist\CalRAG\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Pre-ingested ChromaDB to LocalAppData
Source: "chroma_db\*"; DestDir: "{localappdata}\CalRAG\chroma"; Flags: ignoreversion recursesubdirs createallsubdirs

; Optional: bundled Tesseract
; Source: "vendor\tesseract\*"; DestDir: "{app}\tesseract"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Install Playwright Chromium silently postinstall (non-blocking)
Filename: "{app}\{#MyAppExeName}"; Parameters: "--install-playwright"; Flags: shellexec postinstall runhidden

; Offer to launch app after install
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Keep user data by default
; Type: filesandordirs; Name: "{localappdata}\CalRAG\chroma"
```

---

### Notes

- Streamlit app is the entry point (`streamlit_app.py`). The `windows_launcher.exe` runs Streamlit and opens the browser.
- Data path is always `%LOCALAPPDATA%\CalRAG\chroma`. The app never writes to Program Files.
- `.env` is at `%LOCALAPPDATA%\CalRAG\.env` and is loaded on startup.


