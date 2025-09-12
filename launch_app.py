# Selected Streamlit entry: streamlit_app.py
# To change, set the environment variable ENTRY_REL or edit ENTRY_REL below.

import os, sys
from pathlib import Path

# >>> Set this if your Streamlit entry is not app.py <<<
ENTRY_REL = os.environ.get("ENTRY_REL", r"streamlit_app.py")  # e.g., r"src\\ui.py"

# Optional: load .env with secrets like OPENAI_API_KEY
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

def run_streamlit(entry_rel: str, port: int | None = None):
    from streamlit.web import cli as stcli
    entry = str(Path(__file__).parent / entry_rel)
    argv = ["streamlit", "run", entry, "--server.headless=true"]
    if port:
        argv += ["--server.port", str(port)]
    sys.argv = argv
    raise SystemExit(stcli.main())

if __name__ == "__main__":
    # Allow overriding port by env if 8501 is busy
    port_env = os.getenv("PORT")
    port = int(port_env) if port_env and port_env.isdigit() else None
    run_streamlit(ENTRY_REL, port=port)


