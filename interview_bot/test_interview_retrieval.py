"""Exercise job-focused story retrieval without Streamlit or external API calls."""

import json
from pathlib import Path
import unittest

from interview_retrieval import (
    JOB_FOCUSES,
    build_candidate,
    build_job_context,
    build_search_query,
    clear_candidate_pool,
    coerce_candidate,
    display_category,
    extract_tag_value,
    rank_candidates,
)


QUESTION = "Tell me about a time you solved a difficult problem."


def story_chunk(situation: str, action: str, question: str = QUESTION) -> dict[str, str]:
    """Build a complete story with distinct factual content for retrieval tests."""
    return {
        "text": (
            "Response Type: Story | Principle: Ownership | "
            f"Question: {question} | Situation: {situation} | "
            "Task: I needed to resolve the issue. | "
            f"Action: {action} | Result: The work produced usable recommendations."
        ),
        "response_type": "story",
    }


ROLE_STORIES = [
    story_chunk(
        "I developed a machine learning fault classifier.",
        "I trained XGBoost and evaluated the model using a confusion matrix. "
        "I also built an AI assistant using LLM embeddings and retrieval.",
    ),
    story_chunk(
        "PEC needed an annual transmission planning assessment.",
        "I ran power flow and contingency analysis in PSS/E, evaluated thermal "
        "and voltage violations, and compared transmission upgrades.",
    ),
    story_chunk(
        "A generator needed support through the resource integration process.",
        "I reviewed generation interconnection and full interconnection studies, "
        "RARF registration, reactive testing, and MOD-26 model validation.",
    ),
    story_chunk(
        "Several concurrent projects faced deadlines and budget constraints.",
        "I tracked milestones and the critical path, coordinated stakeholders, "
        "managed scope and dependencies, and prioritized project schedules.",
    ),
]

DIRECT_INTRODUCTION = {
    "text": (
        "Response Type: Direct | Category: Management Profile | "
        "Question: Tell me about yourself. | Aliases: Please introduce yourself. | "
        "Answer: I am an engineer with experience leading technical teams."
    ),
    "response_type": "direct",
}


class JobContextTests(unittest.TestCase):
    def test_default_query_is_preserved_exactly(self) -> None:
        query = "  Tell me about yourself?  "
        self.assertEqual(build_search_query(query), query)
        self.assertEqual(build_job_context("All roles", ""), "")

    def test_selected_role_and_background_enrich_search_query(self) -> None:
        background = "Build retrieval systems for engineering document search."
        result = build_search_query(QUESTION, "AI engineering", background)
        self.assertIn(QUESTION, result)
        self.assertIn("AI engineering", result)
        self.assertIn(background, result)
        self.assertNotEqual(result, build_search_query(QUESTION, "Transmission planning"))
        context = build_job_context("AI engineering", background)
        self.assertIn("AI engineering", context)
        self.assertIn(background, context)

    def test_changing_focus_clears_only_candidate_state(self) -> None:
        messages = [{"role": "user", "content": QUESTION}]
        state = {
            "candidate_stories": [{"text": "Previously selected story"}],
            "candidate_story_pos": 3,
            "active_query": QUESTION,
            "messages": messages,
            "job_focus": "Resource integration",
            "job_background": "Support generation interconnection.",
            "index": "existing-vector-index",
        }
        clear_candidate_pool(state)
        self.assertEqual(state["candidate_stories"], [])
        self.assertEqual(state["candidate_story_pos"], 0)
        self.assertEqual(state["active_query"], "")
        self.assertIs(state["messages"], messages)
        self.assertEqual(state["job_focus"], "Resource integration")
        self.assertEqual(state["job_background"], "Support generation interconnection.")
        self.assertEqual(state["index"], "existing-vector-index")


class FocusedRankingTests(unittest.TestCase):
    def test_same_generic_question_selects_relevant_story_for_each_role(self) -> None:
        expected = {
            "AI engineering": 0,
            "Transmission planning": 1,
            "Resource integration": 2,
            "Project management": 3,
        }
        self.assertTrue(set(expected).issubset(JOB_FOCUSES))
        for focus, expected_index in expected.items():
            with self.subTest(focus=focus):
                results = rank_candidates(
                    QUESTION, ROLE_STORIES, dict.fromkeys(range(4), 0.5), focus=focus
                )
                self.assertTrue(results)
                self.assertEqual(results[0]["index"], expected_index)
                self.assertGreater(results[0]["topic_score"], 0)

    def test_all_roles_preserves_semantic_order_when_question_matches_are_equal(self) -> None:
        results = rank_candidates(
            QUESTION, ROLE_STORIES, {0: 0.1, 1: 0.9, 2: 0.4, 3: 0.3}
        )
        self.assertEqual([item["index"] for item in results], [1, 2, 3, 0])

    def test_ai_does_not_match_substrings_in_ordinary_words(self) -> None:
        ordinary = story_chunk(
            "A colleague said our records were incomplete.",
            "I helped maintain a detailed checklist and repaired the records.",
        )
        results = rank_candidates(QUESTION, [ordinary], {0: 0.99}, focus="AI engineering")
        self.assertEqual(results, [])

    def test_topic_is_supported_by_story_facts_not_question_wording(self) -> None:
        mislabeled = story_chunk(
            "A colleague needed help organizing records.",
            "I reviewed the checklist and resolved missing entries.",
            question="Tell me about a time you solved an AI engineering problem.",
        )
        results = rank_candidates(QUESTION, [mislabeled], {0: 0.99}, focus="AI engineering")
        self.assertEqual(results, [])

    def test_behavioral_query_does_not_fall_back_to_off_topic_story(self) -> None:
        results = rank_candidates(
            QUESTION,
            [ROLE_STORIES[1], DIRECT_INTRODUCTION],
            {0: 0.99, 1: 0.99},
            focus="AI engineering",
        )
        self.assertEqual(results, [])

    def test_general_direct_answer_remains_available_under_a_focus(self) -> None:
        results = rank_candidates(
            "Tell me about yourself.",
            [DIRECT_INTRODUCTION, ROLE_STORIES[0]],
            {0: 0.5, 1: 0.5},
            focus="AI engineering",
        )
        self.assertTrue(results)
        self.assertEqual(results[0]["response_type"], "direct")
        self.assertEqual(results[0]["index"], 0)

    def test_browse_returns_only_matching_stories(self) -> None:
        chunks = [DIRECT_INTRODUCTION, *ROLE_STORIES]
        results = rank_candidates(
            "Browse stories",
            chunks,
            dict.fromkeys(range(len(chunks)), 0.5),
            focus="AI engineering",
            stories_only=True,
        )
        self.assertEqual([item["index"] for item in results], [1])
        self.assertTrue(all(item["response_type"] == "story" for item in results))

    def test_duplicate_story_situations_do_not_fill_answer_pool(self) -> None:
        duplicate = dict(ROLE_STORIES[0])
        duplicate["text"] = duplicate["text"].replace("Principle: Ownership", "Principle: Dive Deep")
        results = rank_candidates(
            QUESTION, [ROLE_STORIES[0], duplicate, ROLE_STORIES[1]], {0: 0.9, 1: 0.8, 2: 0.7}
        )
        self.assertEqual([item["index"] for item in results], [0, 2])
        self.assertEqual(len({item["content_key"] for item in results}), len(results))

    def test_candidate_limit_is_respected(self) -> None:
        results = rank_candidates(QUESTION, ROLE_STORIES, dict.fromkeys(range(4), 0.5), k=2)
        self.assertEqual(len(results), 2)


class ExistingCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path(__file__).with_name("chunks_cleaned.json")
        cls.chunks = json.loads(path.read_text(encoding="utf-8"))

    def test_transmission_browse_includes_annual_assessment(self) -> None:
        results = rank_candidates(
            "Browse stories",
            self.chunks,
            {},
            k=len(self.chunks),
            focus="Transmission planning",
            stories_only=True,
        )
        annual = [item for item in results if "PEC's annual transmission planning assessment internally" in item["text"]]
        self.assertEqual(len(annual), 1)
        self.assertGreater(annual[0]["topic_score"], 0)

    def test_ai_browse_finds_fault_classifier_without_unrelated_annual_assessment(self) -> None:
        results = rank_candidates(
            "Browse stories",
            self.chunks,
            {},
            k=len(self.chunks),
            focus="AI engineering",
            stories_only=True,
        )
        self.assertTrue(any("Power Fault Classifier App" in item["text"] for item in results))
        self.assertFalse(any("PEC's annual transmission planning assessment internally" in item["text"] for item in results))


class CandidateCompatibilityTests(unittest.TestCase):
    def test_story_tags_category_and_coercion_remain_available(self) -> None:
        candidate = build_candidate(0.75, 2, ROLE_STORIES[0])
        self.assertEqual(extract_tag_value(candidate["text"], "Question"), QUESTION)
        self.assertEqual(candidate["response_type"], "story")
        self.assertEqual(display_category(candidate), "Ownership")
        self.assertIs(coerce_candidate(candidate), candidate)
        self.assertEqual(coerce_candidate(ROLE_STORIES[0]["text"])["response_type"], "story")


if __name__ == "__main__":
    unittest.main()
