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
Source: "dist\\CalRAG\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Pre-ingested ChromaDB to LocalAppData
Source: "chroma_db\\*"; DestDir: "{localappdata}\\CalRAG\\chroma"; Flags: ignoreversion recursesubdirs createallsubdirs

; Optional: bundled Tesseract
; Source: "vendor\\tesseract\\*"; DestDir: "{app}\\tesseract"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\\{#MyAppName}"; Filename: "{app}\\{#MyAppExeName}"
Name: "{group}\\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\\{#MyAppName}"; Filename: "{app}\\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Install Playwright Chromium silently postinstall (non-blocking)
Filename: "{app}\\{#MyAppExeName}"; Parameters: "--install-playwright"; Flags: shellexec postinstall runhidden

; Offer to launch app after install
Filename: "{app}\\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Keep user data by default
; Type: filesandordirs; Name: "{localappdata}\\CalRAG\\chroma"
