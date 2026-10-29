"""Reject chatter and unsupported neighbors without imposing a cosine cutoff."""

import unittest

from interview_retrieval import query_validation_message, rank_candidates
from sync_manager_content import serialize_content
from test_interview_retrieval import DIRECT_INTRODUCTION, QUESTION, ROLE_STORIES, story_chunk


CURATED_PLANNING = serialize_content({
    "id": "planning-reconciliation", "response_type": "direct", "category": "Planning",
    "question": "How would you reconcile planning studies?",
    "aliases": ["Explain your planning discrepancy checks."],
    "answer": "I compare transmission planning models and validate the study assumptions.",
    "answer_set": "ercot-planning-20", "question_key": "8 DISCREPANCY",
    "scenario_id": "planning-reconciliation",
})


class QueryValidationTests(unittest.TestCase):
    def test_empty_greetings_and_microphone_checks_request_a_question(self) -> None:
        for query in (
            "", "  ", "...?!", "Hello!", "Hi there.", "Good morning.",
            "Test.", "Testing testing.", "Mic check, one two three.",
            "This is just a test.", "Can you hear me?", "I am testing my microphone.",
            "Check check.",
        ):
            with self.subTest(query=query):
                self.assertIsNotNone(query_validation_message(query, [CURATED_PLANNING]))
                self.assertEqual(rank_candidates(query, [CURATED_PLANNING], {0: 1.0}), [])

    def test_short_authored_and_technical_requests_are_not_rejected(self) -> None:
        for query in (
            "Q8", "8 DISCREPANCY", "DISCREPANCY", "NERC", "PSS/E", "AI", "ML",
            "Contingency analysis?", "Testing protection relays?", "Tell me about yourself.",
            "Tell me about your work.", QUESTION,
        ):
            with self.subTest(query=query):
                self.assertIsNone(query_validation_message(query, [CURATED_PLANNING]))

    def test_explicit_authored_alias_takes_precedence_over_chatter_words(self) -> None:
        chunk = dict(CURATED_PLANNING)
        chunk["text"] = chunk["text"].replace(
            "Aliases: Explain your planning discrepancy checks.", "Aliases: Test",
        )
        self.assertIsNone(query_validation_message("Test.", [chunk]))
        self.assertTrue(rank_candidates("Test.", [chunk], {})[0]["assigned_match"])


class QueryEvidenceTests(unittest.TestCase):
    def test_unrelated_questions_cannot_use_semantic_or_job_scores_as_evidence(self) -> None:
        for query in (
            "How do I bake a sourdough loaf?", "What is the weather today?",
            "안녕하세요 반갑습니다", "Tell me about a time you baked sourdough.",
        ):
            for focus in ("All roles", "Transmission planning"):
                for scores in ({}, {0: -0.25}, {0: 0.99}):
                    with self.subTest(query=query, focus=focus, scores=scores):
                        self.assertEqual(rank_candidates(
                            query, [CURATED_PLANNING], scores, focus=focus,
                            job_description="Lead ERCOT planning and consultant studies.",
                        ), [])

    def test_incidental_location_and_narrative_words_do_not_establish_relevance(self) -> None:
        record = story_chunk(
            "I worked in Austin on a transmission planning assessment.",
            "I checked the study assumptions and compared model results.",
            question="Describe your transmission planning assessment.",
        )
        for query in (
            "How does a restaurant reservation work in Austin?",
            "What is the weather in Austin today?",
        ):
            with self.subTest(query=query):
                self.assertEqual(rank_candidates(query, [record], {0: 0.99}), [])

    def test_substrings_and_source_metadata_do_not_establish_topic_support(self) -> None:
        record = story_chunk(
            "I supported transmission planning equipment testing.",
            "I validated model results.", question="How did you approach equipment testing?",
        )
        record["text"] += " | Source Document: sourdough-baking-test-notes"
        for query in ("Test", "Explain baking sourdough.", "What does the AI do?"):
            with self.subTest(query=query):
                self.assertEqual(rank_candidates(query, [record], {0: 0.99}), [])

    def test_supported_factual_topic_can_win_without_a_positive_semantic_score(self) -> None:
        record = story_chunk(
            "I supported a generator interconnection.",
            "I assessed resource behavior during disturbances using PSS/E.",
            question="Explain your generator assessment.",
        )
        for query, expected_term in (
            ("How do you examine resource behavior during disturbances?", "resource"),
            ("PSS/E", "pss"),
        ):
            with self.subTest(query=query):
                results = rank_candidates(query, [record], {0: -0.25})
                self.assertEqual(len(results), 1)
                self.assertIn(expected_term, results[0]["query_evidence"])

    def test_short_domain_acronyms_remain_whole_word_topics(self) -> None:
        record = story_chunk(
            "I developed an AI classifier and used ML to detect faults.",
            "I evaluated held-out examples.", question="Describe the AI classifier.",
        )
        for query in ("AI", "ML"):
            with self.subTest(query=query):
                self.assertEqual(len(rank_candidates(query, [record], {0: 0.5})), 1)

    def test_broad_interview_prompts_and_role_browsing_preserve_their_scope(self) -> None:
        results = rank_candidates(
            QUESTION, ROLE_STORIES, dict.fromkeys(range(len(ROLE_STORIES)), 0.5),
            focus="Transmission planning",
        )
        self.assertEqual(results[0]["index"], 1)
        self.assertNotIn(0, [item["index"] for item in results])
        self.assertNotIn(3, [item["index"] for item in results])
        introduction = rank_candidates("Tell me about yourself.", [DIRECT_INTRODUCTION], {0: 0.5})
        self.assertTrue(introduction[0]["question_match"])
        browse = rank_candidates(
            "Browse stories", ROLE_STORIES, {}, focus="AI engineering", stories_only=True,
        )
        self.assertEqual([item["index"] for item in browse], [0])

    def test_authored_assignment_remains_independent_of_semantic_neighbors(self) -> None:
        selected = rank_candidates("Q8", [CURATED_PLANNING, ROLE_STORIES[0]], {1: 1.0})
        self.assertEqual([item["source_id"] for item in selected], ["planning-reconciliation"])
        self.assertTrue(selected[0]["assigned_match"])


if __name__ == "__main__":
    unittest.main()
