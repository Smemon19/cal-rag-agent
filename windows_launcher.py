import os
import sys
import time
import webbrowser
import importlib.util
from pathlib import Path


def _set_tesseract_env_if_bundled(app_dir: Path) -> None:
    """If a bundled Tesseract exists under the app dir, point pytesseract to it.

    Looks for tesseract.exe at {app}\tesseract\tesseract.exe and sets TESSERACT_EXE.
    """
    try:
        exe = app_dir / "tesseract" / "tesseract.exe"
        if exe.exists():
            os.environ.setdefault("TESSERACT_EXE", str(exe))
    except Exception:
        pass


def _install_playwright_browsers() -> int:
    """Programmatically install Playwright Chromium browser once.

    Returns process exit code (0 on success).
    """
    try:
        # This mirrors `python -m playwright install chromium`
        from playwright.__main__ import main as pw_main
        # Simulate CLI invocation
        sys.argv = ["playwright", "install", "chromium"]
        pw_main()
        return 0
    except Exception as e:
        # Non-fatal; the app can still run parts that don't require Playwright
        print(f"[postinstall] Playwright install failed: {e}")
        return 1


def _run_streamlit_app(port: int = 8501) -> int:
    """Launch the embedded Streamlit app programmatically on the given port."""
    try:
        # Resolve the real path of streamlit_app.py inside the frozen bundle
        spec = importlib.util.find_spec("streamlit_app")
        if not spec or not spec.origin:
            print("Could not locate streamlit_app module.")
            return 2
        script_path = spec.origin

        # Invoke Streamlit's CLI programmatically
        from streamlit.web.cli import main as st_main
        sys.argv = [
            "streamlit",
            "run",
            script_path,
            "--server.port",
            str(port),
            "--server.headless",
            "false",
        ]
        # Open browser shortly after server starts
        def _open_browser():
            try:
                webbrowser.open(f"http://localhost:{port}")
            except Exception:
                pass

        # Best-effort delay to allow server bind
        # We can't create a perfect race-free detection here; Streamlit itself will auto-open too.
        try:
            import threading
            threading.Timer(1.5, _open_browser).start()
        except Exception:
            pass

        st_main()
        return 0
    except SystemExit as e:
        # Streamlit may sys.exit with code
        return int(getattr(e, "code", 0) or 0)
    except Exception as e:
        print(f"[launcher] Failed to start Streamlit: {e}")
        return 3


def main() -> int:
    # Ensure our writable data dirs and .env exist
    try:
        from utils import ensure_appdata_scaffold
        ensure_appdata_scaffold()
    except Exception:
        # Non-fatal, app may still run
        pass

    # Resolve installation dir (where the .exe lives after PyInstaller)
    app_dir = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent
    os.environ.setdefault("CALRAG_APPDIR", str(app_dir))

    # Wire bundled Tesseract if present
    _set_tesseract_env_if_bundled(app_dir)

    # Handle optional post-install operations
    if "--install-playwright" in sys.argv:
        return _install_playwright_browsers()

    # Default behavior: run Streamlit app
    return _run_streamlit_app(port=8501)


if __name__ == "__main__":
    sys.exit(main())


