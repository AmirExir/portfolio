"""Keep Q1's established and proposed planning paths available without new vectors."""

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock

from interview_evidence import answer_markdown, planning_quick_view
from interview_retrieval import extract_tag_value, rank_candidates
from interview_runtime_content import build_runtime_corpus
from interview_services import render_answer
from sync_manager_content import serialize_content


BASE_DIR = Path(__file__).resolve().parent
PROCESS_ID = "planning-interview-01"
TRANSITION_QUESTIONS = (
    "How does the old ERCOT transmission planning process compare with CTP, PTA, "
    "RPG 2.0, and batch studies?",
    "What is the difference between PTA and RPG 2.0 in the proposed CTP process?",
    "How do batch studies, the commitment window, and the CTP reliability study fit together?",
)


class ProcessTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.content = json.loads((BASE_DIR / "manager_content.json").read_text(encoding="utf-8"))
        cls.saved = json.loads((BASE_DIR / "chunks_cleaned.json").read_text(encoding="utf-8"))
        cls.process = next(item for item in cls.content if item["id"] == PROCESS_ID)
        cls.runtime = build_runtime_corpus(cls.saved, cls.content)

    def test_established_and_proposed_paths_remain_distinct_in_saved_answer(self) -> None:
        """Guard the process distinctions without freezing the surrounding prose."""
        answer = self.process["answer"]
        for concept in (
            "Regional Transmission Plan (RTP)", "Regional Planning Group (RPG)",
            "Comprehensive Transmission Planning (CTP)",
            "Project Transparency and Alternatives (PTA)", "RPG 2.0",
            "standalone Batch → customer commitment → full reliability studies",
        ):
            with self.subTest(concept=concept):
                self.assertIn(concept, answer)
        normalized = answer.casefold()
        for qualifier in ("established", "proposed", "common", "forum", "outside ctp"):
            with self.subTest(qualifier=qualifier):
                self.assertIn(qualifier, normalized)
        self.assertIn("CTP Project Assessment", answer)
        self.assertIn("LCRA", answer)
        self.assertEqual(self.process["response_type"], "direct")
        self.assertNotIn("Preliminary Transmission Assessment", answer)

    def test_transition_questions_select_current_q1_without_semantic_scores(self) -> None:
        """Current aliases must remain useful while Q1's saved vector is stale."""
        queries = (
            "Q1", "PROCESS", "CTP", "PTA", "RPG 2.0", "RPG 2", "Batch study",
            self.process["question"], *TRANSITION_QUESTIONS,
        )
        for query in TRANSITION_QUESTIONS:
            self.assertIn(query, self.process["aliases"])
        for focus in ("All roles", "Transmission planning"):
            for query in queries:
                with self.subTest(focus=focus, query=query):
                    candidates = rank_candidates(query, self.runtime.chunks, {}, focus=focus)
                    self.assertEqual([item["source_id"] for item in candidates], [PROCESS_ID])
                    self.assertTrue(candidates[0]["assigned_match"])
                    self.assertEqual(
                        extract_tag_value(candidates[0]["text"], "Answer"),
                        self.process["answer"],
                    )

    def test_changed_q1_does_not_inherit_its_old_vector_or_modify_saved_corpus(self) -> None:
        """Simulate a prior indexed script and verify only its vector is withheld."""
        prior_content = deepcopy(self.content)
        old_row = next(index for index, item in enumerate(prior_content) if item["id"] == PROCESS_ID)
        prior_content[old_row]["answer"] = "Archived planning-process wording before the transition update."
        saved = [serialize_content(item) for item in prior_content]
        original_saved = deepcopy(saved)

        runtime = build_runtime_corpus(saved, self.content)

        self.assertEqual(saved, original_saved)
        self.assertEqual(runtime.pending_embedding_count, 1)
        self.assertNotIn(old_row, runtime.semantic_row_map)
        self.assertEqual(runtime.remap_scores({old_row: 100.0}), {})
        candidates = rank_candidates(TRANSITION_QUESTIONS[0], runtime.chunks, {}, focus="All roles")
        self.assertEqual([item["source_id"] for item in candidates], [PROCESS_ID])
        self.assertEqual(extract_tag_value(candidates[0]["text"], "Answer"), self.process["answer"])

    def test_saved_render_and_export_preserve_both_paths_and_followup_status(self) -> None:
        """The selected script and its qualifiers must survive display and download."""
        query = TRANSITION_QUESTIONS[1]
        candidate = rank_candidates(query, self.runtime.chunks, {}, focus="Transmission planning")[0]
        client = Mock()
        rendered = render_answer(client, query, candidate, "Transmission planning interview")
        expected = self.process["answer"] + "\n\n**Follow-up preparation**\n\n" + self.process["reflection"]
        self.assertEqual(rendered, expected)
        client.chat.completions.create.assert_not_called()
        exported = answer_markdown({"question": query, "text": rendered, "candidate": candidate})
        self.assertIn(expected, exported)
        self.assertIn("Saved answer wording", exported)

    def test_recall_cue_and_reflection_keep_transition_and_status_visible(self) -> None:
        """Navigation and follow-up preparation must not hide proposal qualifiers."""
        row = next(row for row in planning_quick_view(self.runtime.chunks) if row["source_id"] == PROCESS_ID)
        for concept in ("CTP", "PTA", "RPG 2.0", "Batch"):
            with self.subTest(concept=concept):
                self.assertIn(concept, row["cue"])
        self.assertRegex(row["cue"].casefold(), r"\b(?:proposed|developing)\b")
        reflection = self.process["reflection"].casefold()
        self.assertIn("effective", reflection)
        self.assertRegex(reflection, r"\b(?:proposed|proposal|proposals)\b")

    def test_november_design_does_not_redate_the_historical_rule_register(self) -> None:
        """A newer workshop must not imply renewed verification of older rule statuses."""
        answer = self.process["answer"]
        proposal = answer.split("**CTP —", 1)[1].split("**Keep a Dated Register", 1)[0]
        for qualifier in (
            "November 17, 2026", "preliminary study-duration targets",
            "not effective deadlines", "for CTP purposes", "both assessment pathways",
        ):
            with self.subTest(qualifier=qualifier):
                self.assertIn(qualifier, proposal)
        register = answer.split("**Keep a Dated Register", 1)[1].split("**5.", 1)[0]
        for qualifier in ("Historical reference", "October 21, 2026", "October 22, 2026",
                          "not a November status update"):
            self.assertIn(qualifier, register)
        references = "\n".join(self.process["references"])
        for source in (
            "CTP-Key-Design-Principles-Whitepaper-draft_v06_EXTERNAL.docx",
            "ERCOT-CTP-workshop_4a_20261117.pptx",
        ):
            self.assertIn(source, references)


if __name__ == "__main__":
    unittest.main()
