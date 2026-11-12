#!/usr/bin/env python3
"""Start the explorer from a fresh copy: creates .venv next to this file, installs requirements.txt into it and runs
Streamlit. Anything after -- is passed to `streamlit run` (e.g. -- --server.port 8502)."""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENV = os.path.join(HERE, ".venv")
PY = os.path.join(VENV, "Scripts", "python.exe") if os.name == "nt" else os.path.join(VENV, "bin", "python")

if sys.version_info < (3, 10):
    sys.exit("Python 3.10 or newer is required")
if not os.path.exists(PY):
    print("[run_dashboard] creating .venv and installing the requirements (first run only) ...", flush=True)
    subprocess.run([sys.executable, "-m", "venv", VENV], check=True)
    subprocess.run([PY, "-m", "pip", "install", "--quiet", "--upgrade", "pip"], check=True)
    subprocess.run([PY, "-m", "pip", "install", "--quiet", "-r", os.path.join(HERE, "requirements.txt")], check=True)
extra = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
print("[run_dashboard] starting at http://localhost:8501 (Ctrl+C to stop)", flush=True)
sys.exit(subprocess.call([PY, "-m", "streamlit", "run", os.path.join(HERE, "app.py"), "--browser.gatherUsageStats", "false", *extra], cwd=HERE))
