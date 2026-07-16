"""Legacy filename retained as an alias of the centralized Resource Integration assistant."""

import runpy
from pathlib import Path


runpy.run_path(str(Path(__file__).with_name("ercotAIassistant.py")), run_name="__main__")
