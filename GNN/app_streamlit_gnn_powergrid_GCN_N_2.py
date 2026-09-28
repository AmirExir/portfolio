"""Compatibility launcher for the shared GNN scenario application."""
from pathlib import Path
import runpy
import sys

app_directory = Path(__file__).resolve().parent
if str(app_directory) not in sys.path:
    sys.path.insert(0, str(app_directory))
runpy.run_path(str(app_directory / "app_gnn_streamlit.py"), run_name="__main__")
