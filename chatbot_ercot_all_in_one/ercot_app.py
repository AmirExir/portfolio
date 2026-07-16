"""Compatibility entry point for the centralized all-in-one ERCOT assistant."""

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).with_name("ercot_assistant_app.py")), run_name="__main__")
