"""Keep the uncertainty recommendation answer focused on transmission planning."""

import json
from pathlib import Path
import unittest
from unittest.mock import Mock

import test_interview_ui as ui_fixtures
from interview_retrieval import build_candidate, extract_tag_value, planning_evidence
from interview_services import answer_messages
from sync_manager_content import serialize_content


BASE_DIR = Path(__file__).resolve().parent


def planning_recommendation_source() -> dict:
    """Read the current managed Q14 record without relying on saved embeddings."""
    content = json.loads((BASE_DIR / "manager_content.json").read_text(encoding="utf-8"))
    return next(item for item in content if item["id"] == "planning-interview-14")


def explicit_planning_question(source: dict) -> str:
    """Reproduce the user's transmission-planning version of the saved question."""
    return source["question"].replace(
        "Describe a situation where", "Describe a situation in transmission planning where", 1,
    )


class Q14PlanningPromptTests(unittest.TestCase):
    def test_planning_source_preserves_user_scenario_and_prospective_wording(self) -> None:
        source = planning_recommendation_source()
        question = explicit_planning_question(source)
        self.assertIn(question, source["aliases"])
        chunk = serialize_content(source)
        self.assertTrue(planning_evidence(chunk["text"]))
        self.assertEqual(source["response_type"], "direct")
        self.assertIn("minimax regret", source["answer"].casefold())
        messages = answer_messages(question, build_candidate(1, 0, chunk), "")
        instructions = messages[0]["content"]
        self.assertIn("without forcing STAR", instructions)
        self.assertIn("proposed approaches ('I would')", instructions)
        self.assertIn(chunk["text"], messages[1]["content"])


@unittest.skipIf(ui_fixtures.AppTest is None, "Streamlit and NumPy are required for UI tests")
class Q14PlanningUITests(unittest.TestCase):
    setUp = ui_fixtures.JobFocusUITests.setUp
    apply_focus = ui_fixtures.JobFocusUITests.apply_focus

    def _check_question_variants(self, focus: str) -> None:
        """Favor the current planning answer despite high saved EMS similarities."""
        source = planning_recommendation_source()
        expected = serialize_content(source)["text"]
        self.apply_focus(focus)
        saved = json.loads((BASE_DIR / "chunks_cleaned.json").read_text(encoding="utf-8"))
        ems_rows = [
            row for row, chunk in enumerate(saved)
            if any(phrase in chunk["text"].lower() for phrase in (
                "ge energy management", "ge ems", "existing ge energy",
            ))
        ]
        self.assertTrue(ems_rows, "The regression needs the previous EMS evidence in the saved index")
        self.app.session_state["index"].search = Mock(return_value=(
            ui_fixtures.np.array([[0.99] * len(ems_rows)]),
            ui_fixtures.np.array([ems_rows]),
        ))
        planning_question = explicit_planning_question(source)
        questions = (
            source["question"],
            planning_question,
            f"14.\u2060 \u2060{planning_question} 15",
        )
        for question in questions:
            with self.subTest(focus=focus, question=question):
                self.app.chat_input[0].set_value(question).run()
                self.assertFalse(self.app.exception)
                selected = self.app.session_state["active_answer"]["candidate"]
                self.assertEqual(selected["text"], expected)
                self.assertTrue(selected["question_match"])
                self.assertTrue(planning_evidence(selected["text"]))
                answer = extract_tag_value(selected["text"], "Answer")
                for detail in (
                    "minimax regret", "PSCAD", "PLEXOS", "750 MW", "STATCOM", "HTLS",
                    "**1. Isolating", "**2. Engineering", "**3. Quantifying", "**4. The Final",
                ):
                    self.assertIn(detail.casefold(), answer.casefold())
                self.assertNotRegex(answer, r"(?i)\b(?:GE|EMS|SCADA)\b|energy management system")
                displayed = self.app.session_state["active_answer"]["text"]
                heading = "\n\n**Follow-up preparation**\n\n"
                self.assertEqual(displayed.count(heading), 1)
                script, reflection = displayed.split(heading)
                self.assertEqual(script, source["answer"])
                self.assertEqual(reflection, source["reflection"])
                self.assertEqual(len(self.app.session_state["candidate_stories"]), 1)
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()

    def test_all_roles_prefers_current_planning_answer_for_exact_and_numbered_questions(self) -> None:
        self._check_question_variants("All roles")

    def test_planning_focus_prefers_current_planning_answer_for_exact_and_numbered_questions(self) -> None:
        self._check_question_variants("Transmission planning")


if __name__ == "__main__":
    unittest.main()
