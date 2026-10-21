"""Verify the planning cheat sheet preserves assigned answers and evidence gates."""

from copy import deepcopy
import json
from pathlib import Path
import unittest

from embedding_utils import plan_embedding_reuse
from interview_evidence import planning_quick_view, story_id
from sync_manager_content import serialize_content
import test_interview_ui as ui_fixtures


BASE_DIR = Path(__file__).resolve().parent
QUESTION_KEYS = [
    "1 PROCESS", "2 VIOLATION", "3 NERC", "4 GENERATOR", "5 DATA CENTER",
    "6 IBR", "7 ECONOMICS", "8 DISCREPANCY", "9 STUDY TYPES", "10 SCENARIOS",
    "11 PRIORITIZE INVESTMENTS", "12 TOOLS", "13 LEADERSHIP", "14 UNCERTAINTY",
    "15 DISAGREEMENT", "16 DEADLINES", "17 BAD NEWS", "18 ERROR",
    "19 MENTORING", "20 STAKEHOLDERS",
]


def managed_content() -> list[dict]:
    """Read the same authored content the running application overlays."""
    return json.loads((BASE_DIR / "manager_content.json").read_text(encoding="utf-8"))


class PlanningQuickViewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.content = managed_content()
        self.primary = [
            item for item in self.content if item.get("answer_set") == "ercot-planning-20"
        ]

    def test_all_twenty_keywords_have_unique_sorted_primary_recall_rows(self) -> None:
        chunks = [serialize_content(item) for item in reversed(self.content)]
        rows = planning_quick_view(chunks)
        self.assertEqual([row["question_key"] for row in rows], QUESTION_KEYS)
        self.assertEqual(len({row["source_id"] for row in rows}), 20)
        self.assertEqual(len({row["title"] for row in rows}), 20)
        self.assertEqual(len({row["cue"] for row in rows}), 20)
        source_by_id = {item["id"]: item for item in self.primary}
        for row in rows:
            with self.subTest(keyword=row["question_key"]):
                self.assertEqual(set(row), {"source_id", "question_key", "title", "cue", "basis"})
                source = source_by_id[row["source_id"]]
                self.assertEqual(row["question_key"], source["question_key"])
                for field in ("title", "cue", "basis"):
                    self.assertEqual(row[field], source["quick_view"][field])
                    self.assertTrue(row[field].strip())
        self.assertNotIn("planning-backup-15-ratings", [row["source_id"] for row in rows])

    def test_illustrative_examples_are_labeled_separately_from_personal_experience(self) -> None:
        rows = {row["question_key"]: row for row in planning_quick_view(
            [serialize_content(item) for item in self.primary]
        )}
        for key in ("14 UNCERTAINTY", "20 STAKEHOLDERS"):
            with self.subTest(keyword=key):
                self.assertEqual(rows[key]["basis"], "Illustrative scenario")
        for key in ("15 DISAGREEMENT", "18 ERROR"):
            with self.subTest(keyword=key):
                self.assertEqual(rows[key]["basis"], "Experience")

    def test_recall_metadata_does_not_rewrite_answers_or_require_new_embeddings(self) -> None:
        before = [serialize_content({key: value for key, value in item.items() if key != "quick_view"})
                  for item in self.primary]
        after = [serialize_content(item) for item in self.primary]
        self.assertEqual([chunk["text"] for chunk in before], [chunk["text"] for chunk in after])
        self.assertEqual(plan_embedding_reuse(before, after), (list(range(20)), []))
        for item, chunk in zip(self.primary, after):
            with self.subTest(source=item["id"]):
                self.assertEqual(chunk["quick_view"], item["quick_view"])
        self.assertEqual(planning_quick_view(before), [])

    def test_malformed_quick_view_metadata_is_rejected(self) -> None:
        record = deepcopy(self.primary[0])
        valid = {"title": "Project development", "cue": "Need to delivery", "basis": "Proposed approach"}
        malformed = [None, "summary", [], {}, {"title": "Title"}, {**valid, "extra": "unexpected"}]
        for field in ("title", "cue", "basis"):
            malformed.extend({**valid, field: invalid} for invalid in (None, 10, "", "  "))
        malformed.append({**valid, "basis": "Verified experience"})
        for metadata in malformed:
            with self.subTest(metadata=metadata):
                with self.assertRaises(ValueError):
                    serialize_content({**record, "quick_view": metadata})

    def test_quick_view_is_only_for_primary_planning_assignments(self) -> None:
        metadata = self.primary[0]["quick_view"]
        backup = next(item for item in self.content if item.get("backup_for"))
        ordinary = next(item for item in self.content if not item.get("answer_set") and not item.get("backup_for"))
        other_set = {**self.primary[0], "answer_set": "other-question-set"}
        for record in (backup, ordinary, other_set):
            with self.subTest(source=record["id"], answer_set=record.get("answer_set")):
                with self.assertRaises(ValueError):
                    serialize_content({**record, "quick_view": metadata})
        legacy = {key: value for key, value in self.primary[0].items() if key != "quick_view"}
        self.assertNotIn("quick_view", serialize_content(legacy))


@unittest.skipIf(ui_fixtures.AppTest is None, "Streamlit and NumPy are required for UI tests")
class PlanningQuickViewUITests(unittest.TestCase):
    setUp = ui_fixtures.JobFocusUITests.setUp
    click_button = ui_fixtures.JobFocusUITests.click_button

    def record(self, key: str) -> dict:
        """Get the exact saved script assigned to one keyword."""
        return next(item for item in managed_content() if item.get("question_key") == key)

    def choose(self, key: str) -> None:
        """Select a keyword through the visible control without submitting a prompt."""
        self.app.selectbox(key="planning_quick_view_question").select(key).run()
        self.assertFalse(self.app.exception)

    def assert_saved_answer(self, record: dict) -> None:
        """Check the normal saved answer and follow-up path remains intact."""
        active = self.app.session_state["active_answer"]
        self.assertEqual(active["candidate"]["source_id"], record["id"])
        self.assertEqual(active["text"], record["answer"] + "\n\n**Follow-up preparation**\n\n" + record["reflection"])

    def test_cheat_sheet_shows_keywords_cues_and_basis_without_api_calls(self) -> None:
        self.assertTrue(any(item.label == "Planning interview cheat sheet" for item in self.app.expander))
        selector = self.app.selectbox(key="planning_quick_view_question")
        self.assertEqual(selector.label, "Choose a keyword")
        self.assertEqual([option.split(" · ", 1)[0] for option in selector.options], QUESTION_KEYS)
        table = next(frame.value for frame in self.app.dataframe if "Recall cue" in frame.value.columns)
        self.assertEqual(list(table.columns), ["Shortcut", "Story / approach", "Recall cue", "Basis"])
        self.assertEqual(table["Shortcut"].tolist(), QUESTION_KEYS)
        self.assertEqual(table.set_index("Shortcut").loc["14 UNCERTAINTY", "Basis"], "Illustrative scenario")
        self.assertEqual(table.set_index("Shortcut").loc["20 STAKEHOLDERS", "Basis"], "Illustrative scenario")
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()

    def test_selected_keywords_open_exact_saved_answers_with_followups(self) -> None:
        for key in ("15 DISAGREEMENT", "18 ERROR"):
            with self.subTest(keyword=key):
                self.choose(key)
                self.click_button("Show mapped answer")
                self.assert_saved_answer(self.record(key))
                self.assertEqual(self.app.session_state["active_query"], key)
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()

    def test_quick_view_honors_correction_gate_without_substituting_backup(self) -> None:
        self.choose("18 ERROR")
        self.click_button("Show mapped answer")
        self.assertIn("active_answer", self.app.session_state)
        source = self.record("15 DISAGREEMENT")
        self.app.session_state["story_reviews"] = {
            story_id(serialize_content(source)["text"]): {"review_status": "Needs correction"},
        }
        self.choose("15 DISAGREEMENT")
        table = next(frame.value for frame in self.app.dataframe if "Recall cue" in frame.value.columns)
        row = table.set_index("Shortcut").loc["15 DISAGREEMENT"]
        self.assertEqual(row["Story / approach"], "Needs correction · " + source["quick_view"]["title"])
        self.assertEqual(row["Recall cue"], "Review this story before rehearsing.")
        self.assertNotIn(source["quick_view"]["cue"], table["Recall cue"].tolist())
        self.assertFalse(any(source["quick_view"]["cue"] in item.value for item in self.app.markdown))
        self.assertTrue(any(
            "You marked this story as needing correction." in warning.value
            for warning in self.app.warning
        ))
        self.click_button("Show mapped answer")
        self.assertNotIn("active_answer", self.app.session_state)
        self.assertEqual(self.app.session_state["candidate_stories"], [])
        self.client.embeddings.create.assert_not_called()
        self.client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
