"""Check timer installation, dashboard routing, and preservation of existing delivery branches."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
import shlex
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from market_agent.breaking_analysis_workflow import ANALYSIS_NODE, POLL_NODE, repair_breaking_analysis_workflow
from market_agent.publisher_headlines import HeadlineCheck


class BreakingAnalysisWorkflowTests(unittest.TestCase):
    def test_timer_is_isolated_and_uses_the_configured_checkout(self) -> None:
        graph = {
            "nodes": [{"name": "News API"}, {"name": "Run Optimization Scripts"}],
            "connections": {"News API": {"main": [[{"node": "Existing brief", "type": "main", "index": 0}]]}},
        }
        original = copy.deepcopy(graph)
        root = Path("/opt/market agent's checkout")
        installed, changes = repair_breaking_analysis_workflow(graph, project_root=root)
        self.assertTrue(changes)
        self.assertEqual(graph, original)
        self.assertEqual(installed["connections"]["News API"], original["connections"]["News API"])
        self.assertEqual(installed["connections"][POLL_NODE]["main"][0][0]["node"], ANALYSIS_NODE)
        self.assertNotIn(ANALYSIS_NODE, installed["connections"])
        timer, runner = installed["nodes"][-2:]
        self.assertEqual(timer["parameters"]["rule"]["interval"], [{"field": "minutes", "minutesInterval": 5}])
        self.assertEqual(shlex.split(runner["parameters"]["command"]),
                         ["cd", str(root), "&&", "./.venv/bin/python", "-m", "market_agent.breaking_analysis"])
        self.assertEqual(repair_breaking_analysis_workflow(installed, project_root=root), (installed, ()))
        installed["connections"][POLL_NODE]["main"][0].append({"node": "Another destination"})
        with self.assertRaisesRegex(ValueError, "unexpected destinations"):
            repair_breaking_analysis_workflow(installed, project_root=root)

    def test_dashboard_shows_source_and_daily_session_of_completed_run(self) -> None:
        report = {
            "generated_at": "2026-11-23T15:00:00Z",
            "events": [{"title": "Board appoints CEO", "url": "https://example.org/news",
                        "source": "Fixture", "published_at": "2026-11-23T14:00:00Z"}],
            "as_of_sessions": ["2026-11-20"], "summary": "Fixture daily forecast", "errors": [],
        }
        script = (
            "from pathlib import Path\n"
            "from market_agent.breaking_analysis_view import analyze_refreshed_headlines, render_breaking_analysis\n"
            "render_breaking_analysis(Path('fixture'))"
        )
        with patch("market_agent.breaking_analysis_view.load_latest_analysis", return_value=report):
            app = AppTest.from_string(script).run()
        self.assertFalse(app.exception)
        self.assertIn("Nov 20, 2026", app.caption[-1].value)
        self.assertEqual(app.text[0].value, report["summary"])
        self.assertEqual(app.get("link_button")[0].proto.url, report["events"][0]["url"])

    def test_refresh_hands_the_same_publisher_check_to_the_poller(self) -> None:
        from market_agent.breaking_analysis_view import analyze_refreshed_headlines

        check = HeadlineCheck(datetime.now(timezone.utc), (), (), ())
        with patch("market_agent.breaking_analysis_view.poll_breaking_analysis", return_value={"status": "no_event"}) as poll:
            analyze_refreshed_headlines(Path("fixture"), check)
        self.assertIs(poll.call_args.kwargs["check_feeds"](), check)
        self.assertEqual(poll.call_args.kwargs["now"], check.checked_at)
