"""Validate job-description context and ranking without external services."""

import unittest

from interview_retrieval import (
    MAX_JOB_DESCRIPTION_CHARS,
    MAX_JOB_SEARCH_CHARS,
    build_job_context,
    build_search_query,
    job_description_excerpt,
    rank_candidates,
)
from test_interview_retrieval import DIRECT_INTRODUCTION, QUESTION, ROLE_STORIES, story_chunk


LATE_REQUIREMENT = (
    "Required qualifications: Perform transmission planning in TARA and ASPEN OneLiner, "
    "evaluate thermal and voltage violations, and compare upgrade alternatives."
)
LONG_JOB_DESCRIPTION = (
    "Project Director, ERCOT Transmission Planning\n"
    + "Company information: Browse our offices and employee benefits on our careers website.\n" * 65
    + LATE_REQUIREMENT
)


class JobDescriptionContextTests(unittest.TestCase):
    def test_full_description_is_retained_for_generation(self) -> None:
        self.assertGreater(len(LONG_JOB_DESCRIPTION), MAX_JOB_SEARCH_CHARS)
        context = build_job_context(job_description=LONG_JOB_DESCRIPTION)
        self.assertIn(LONG_JOB_DESCRIPTION, context)
        self.assertIn(LATE_REQUIREMENT, context)

    def test_bounded_search_excerpt_includes_requirement_near_end(self) -> None:
        excerpt = job_description_excerpt(LONG_JOB_DESCRIPTION, QUESTION)
        self.assertLessEqual(len(excerpt), MAX_JOB_SEARCH_CHARS)
        self.assertIn("TARA", excerpt)
        self.assertIn("ASPEN OneLiner", excerpt)
        query = build_search_query(QUESTION, job_description=LONG_JOB_DESCRIPTION)
        self.assertIn(QUESTION, query)
        self.assertIn("TARA", query)
        self.assertNotIn(LONG_JOB_DESCRIPTION, query)

    def test_job_description_alone_guides_each_question_with_all_roles(self) -> None:
        description = "Responsibilities: Build machine learning classifiers and evaluate XGBoost models."
        for question in (QUESTION, "How do you evaluate a model?"):
            with self.subTest(question=question):
                query = build_search_query(question, job_description=description)
                self.assertIn(question, query)
                self.assertIn("XGBoost", query)
                self.assertNotEqual(query, question)

    def test_maximum_accepted_description_is_not_truncated_for_generation(self) -> None:
        description = "x" * (MAX_JOB_DESCRIPTION_CHARS - 11) + "END_MARKER!"
        self.assertEqual(len(description), MAX_JOB_DESCRIPTION_CHARS)
        self.assertIn(description, build_job_context(job_description=description))

    def test_oversized_description_is_rejected_instead_of_silently_cut(self) -> None:
        description = "x" * (MAX_JOB_DESCRIPTION_CHARS + 1)
        builders = (
            lambda: build_job_context(job_description=description),
            lambda: build_search_query(QUESTION, job_description=description),
            lambda: job_description_excerpt(description, QUESTION),
        )
        for build in builders:
            with self.subTest(builder=build):
                with self.assertRaises(ValueError):
                    build()

    def test_empty_description_preserves_default_query(self) -> None:
        query = "  Tell me about yourself?  "
        self.assertEqual(build_search_query(query, job_description=""), query)
        self.assertEqual(build_job_context(job_description=""), "")
        self.assertEqual(job_description_excerpt("", query), "")


class JobDescriptionRankingTests(unittest.TestCase):
    def test_job_description_reorders_generic_answers_without_manual_focus(self) -> None:
        chunks = [ROLE_STORIES[1], ROLE_STORIES[0]]
        description = "Build machine learning classifiers using XGBoost and LLM embeddings."
        results = rank_candidates(
            QUESTION, chunks, {0: 0.5, 1: 0.5}, job_description=description
        )
        self.assertEqual(results[0]["index"], 1)

    def test_explicit_manual_focus_still_filters_conflicting_description(self) -> None:
        results = rank_candidates(
            QUESTION,
            [ROLE_STORIES[1], ROLE_STORIES[0]],
            {0: 0.5, 1: 0.99},
            focus="Transmission planning",
            job_description="Build machine learning classifiers using XGBoost and LLM embeddings.",
        )
        self.assertEqual([item["index"] for item in results], [0])

    def test_description_does_not_override_an_exact_direct_introduction(self) -> None:
        results = rank_candidates(
            "Tell me about yourself.",
            [DIRECT_INTRODUCTION, ROLE_STORIES[0]],
            {0: 0.5, 1: 0.5},
            job_description="Build machine learning classifiers using XGBoost and LLM embeddings.",
        )
        self.assertEqual(results[0]["response_type"], "direct")

    def test_repeated_job_terms_do_not_inflate_candidate_score(self) -> None:
        description = "machine learning XGBoost LLM embeddings"
        chunks = [ROLE_STORIES[0]]
        once = rank_candidates(QUESTION, chunks, {0: 0.5}, job_description=description)
        repeated = rank_candidates(
            QUESTION, chunks, {0: 0.5}, job_description=(description + "\n") * 100
        )
        self.assertEqual(once[0]["score"], repeated[0]["score"])

    def test_question_only_job_keywords_are_not_treated_as_experience(self) -> None:
        chunk = story_chunk(
            "A colleague needed help with missing records.",
            "I reviewed a checklist and resolved missing entries.",
            question="Describe your machine learning experience with XGBoost.",
        )
        baseline = rank_candidates(QUESTION, [chunk], {0: 0.5})
        with_description = rank_candidates(
            QUESTION, [chunk], {0: 0.5}, job_description="machine learning XGBoost"
        )
        self.assertEqual(with_description[0]["score"], baseline[0]["score"])


if __name__ == "__main__":
    unittest.main()
