"""Launcher weather defaults do not contact sources or select a new model."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(shutil.which("zsh"), "The installed launcher uses zsh")
class ForecastLauncherTests(unittest.TestCase):
    def test_collect_default_and_explicit_weather_overrides(self) -> None:
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as folder:
            executable = Path(folder) / "fake-python"
            executable.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "print(json.dumps({'mode': os.environ['ERCOT_WEATHER_MODE'], "
                "'agent': os.environ['ERCOT_WEATHER_USER_AGENT'], 'args': sys.argv[1:]}))\n"
            )
            executable.chmod(0o700)
            for mode in (None, "off", "model"):
                with self.subTest(mode=mode):
                    environment = dict(os.environ)
                    environment.pop("ERCOT_WEATHER_MODE", None)
                    environment.pop("ERCOT_WEATHER_USER_AGENT", None)
                    environment.update(ERCOT_FORECAST_PYTHON=str(executable), ERCOT_REPO_ROOT=str(root))
                    if mode:
                        environment.update(ERCOT_WEATHER_MODE=mode, ERCOT_WEATHER_USER_AGENT="custom-weather-application")
                    result = subprocess.run(
                        ["zsh", str(root / "scripts/run_ercot_load_forecast.sh"), "--compare"],
                        env=environment, text=True, capture_output=True, check=True, timeout=10,
                    )
                    output = json.loads(result.stdout)
                    self.assertEqual(output["mode"], mode or "collect")
                    self.assertEqual(output["agent"], "custom-weather-application" if mode else "ercot-load-forecast/1.0")
                    self.assertEqual(output["args"], ["-m", "ERCOTAPI.load_forecast_report", "--compare"])


if __name__ == "__main__":
    unittest.main()
