"""Compatibility entrypoint for running the market agent from the repo root.

Use either:
    streamlit run market_agent_app.py
    streamlit run market_agent/market_agent_app.py
"""

from pathlib import Path
import runpy


APP_PATH = Path(__file__).resolve().parent / "market_agent" / "market_agent_app.py"
runpy.run_path(str(APP_PATH), run_name="__main__")
