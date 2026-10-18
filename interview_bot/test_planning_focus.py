"""Keep planning-focused retrieval grounded in study purpose, not shared tools."""

import json
from pathlib import Path
import unittest

from interview_retrieval import extract_tag_value, rank_candidates, topic_score


REPORTED_QUESTION = (
    "Tell me about a time when you took on something significant outside your "
    "area of responsibility. Why was it important? What was the outcome?"
)
PLANNING_JOB = (
    "Lead ERCOT transmission planning, generator and large-load interconnection "
    "studies, power flow and contingency analysis using PSS/E and TARA. "
    "Compare transmission alternatives and present investment recommendations."
)


def synthetic_story(situation: str, action: str, question: str = REPORTED_QUESTION) -> dict[str, str]:
    """Create a complete STAR record without adding implicit planning evidence."""
    return {
        "response_type": "story",
        "text": (
            f"Response Type: Story | Principle: Ownership | Question: {question} | "
            f"Situation: {situation} | Task: I took ownership of the work. | "
            f"Action: {action} | Result: The team completed the work successfully."
        ),
    }


class PlanningPurposeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.chunks = json.loads(Path(__file__).with_name("chunks_cleaned.json").read_text(encoding="utf-8"))

    def saved_story(self, situation_prefix: str) -> dict:
        """Find actual source evidence by its situation rather than array order."""
        matches = [chunk for chunk in self.chunks if extract_tag_value(chunk["text"], "Situation").startswith(situation_prefix)]
        self.assertEqual(len(matches), 1, f"Expected one saved story starting with {situation_prefix!r}")
        return matches[0]

    def ems_story(self) -> dict:
        """Return the EMS upgrade record selected in the reported failure."""
        return self.saved_story(
            "At LCRA, we had been using a GE Energy Management System (EMS) "
            "for several years to monitor and control grid operations."
        )

    def annual_assessment(self) -> dict:
        """Return PEC's saved annual-assessment story."""
        return self.saved_story("When I joined PEC, our annual transmission planning assessment")

    def test_reported_question_excludes_ems_even_with_stronger_semantic_match(self) -> None:
        ems, annual = self.ems_story(), self.annual_assessment()
        for description in ("", PLANNING_JOB):
            with self.subTest(job_description=bool(description)):
                results = rank_candidates(
                    REPORTED_QUESTION, [ems, annual], {0: 0.99, 1: 0.1},
                    focus="Transmission planning", job_description=description,
                )
                self.assertEqual([candidate["index"] for candidate in results], [1])

    def test_shared_analysis_and_software_names_do_not_qualify_ems_testing(self) -> None:
        ems = synthetic_story(
            "Our GE EMS needed an upgrade for real-time control-room operations.",
            "I tested SCADA and state estimator applications through factory and "
            "site acceptance testing. I reviewed power flow, load flow, steady-state, "
            "dynamic stability, contingency analysis, PSS/E, and TARA applications "
            "against operator workflows and coordinated software bug fixes with the vendor.",
        )
        self.assertEqual(topic_score(ems["text"], "Transmission planning"), 0)
        self.assertEqual(rank_candidates(
            REPORTED_QUESTION, [ems], {0: 0.99}, focus="Transmission planning",
            job_description=PLANNING_JOB,
        ), [])

    def test_planning_question_and_aliases_do_not_supply_evidence(self) -> None:
        record = synthetic_story(
            "Our administrative records had missing entries.",
            "I reviewed handover checklists and corrected the record ownership.",
            question="Describe a transmission planning assessment and transmission upgrade study.",
        )
        record["text"] = record["text"].replace(
            " | Situation:",
            " | Aliases: Explain your planning models, PSS/E studies, and TARA contingency analysis. | Situation:",
        )
        self.assertEqual(topic_score(record["text"], "Transmission planning"), 0)
        self.assertEqual(rank_candidates(
            REPORTED_QUESTION, [record], {0: 0.99}, focus="Transmission planning",
        ), [])

    def test_actual_planning_studies_remain_eligible(self) -> None:
        # These cover studies, automation, and investment planning, including
        # evidence that never uses the exact phrase "transmission planning".
        situations = (
            "As part of our steady-state N-1 contingency analysis process,",  # P3/P6 automation
            "A principal engineer planned to install seven capacitor banks",  # interconnection alternatives
            "At LCRA, the surge in interconnection requests from data centers, renewable projects, and cryptocurrency facilities",  # model automation
            "At LCRA, five-year load forecasts showed significant growth",  # investment alternatives
            "When I joined PEC, our annual transmission planning assessment",  # annual assessment
        )
        for prefix in situations:
            with self.subTest(situation=prefix):
                record = self.saved_story(prefix)
                self.assertGreater(topic_score(record["text"], "Transmission planning"), 0)
                results = rank_candidates(
                    REPORTED_QUESTION, [record], {0: 0.1}, focus="Transmission planning",
                )
                self.assertEqual([candidate["index"] for candidate in results], [0])

    def test_operational_stories_do_not_fill_planning_answer_or_browse_pool(self) -> None:
        records = [
            self.ems_story(),
            self.saved_story("I was responsible for preparing and facilitating the Black Start simulation,"),
            self.saved_story("Early in my career, during a period of high workload at ERCOT,"),
        ]
        for record in records:
            with self.subTest(situation=extract_tag_value(record["text"], "Situation")):
                self.assertEqual(topic_score(record["text"], "Transmission planning"), 0)
        for query, browse in ((REPORTED_QUESTION, False), ("Browse stories", True)):
            with self.subTest(browse=browse):
                self.assertEqual(rank_candidates(
                    query, records, dict.fromkeys(range(len(records)), 0.99),
                    focus="Transmission planning", stories_only=browse,
                ), [])

    def test_planning_rating_validation_is_not_rejected_for_mentioning_ems(self) -> None:
        record = self.saved_story("During a pre-audit review, I discovered a major discrepancy between equipment ratings")
        self.assertIn("real-time EMS model", record["text"])
        self.assertGreater(topic_score(record["text"], "Transmission planning"), 0)
        results = rank_candidates(REPORTED_QUESTION, [record], {0: 0.5}, focus="Transmission planning")
        self.assertEqual([candidate["index"] for candidate in results], [0])

    def test_ems_remains_available_for_all_roles_and_project_management(self) -> None:
        for focus in ("All roles", "Project management"):
            with self.subTest(focus=focus):
                results = rank_candidates(REPORTED_QUESTION, [self.ems_story()], {0: 0.9}, focus=focus)
                self.assertEqual([candidate["index"] for candidate in results], [0])

    def test_direct_introduction_remains_available_under_planning_focus(self) -> None:
        introduction = next(chunk for chunk in self.chunks if extract_tag_value(
            chunk["text"], "Question"
        ) == "Please introduce yourself for this management position.")
        results = rank_candidates(
            "Please introduce yourself for this management position.",
            [self.annual_assessment(), introduction], {0: 0.5, 1: 0.5},
            focus="Transmission planning", job_description=PLANNING_JOB,
        )
        self.assertTrue(results)
        self.assertEqual(results[0]["index"], 1)
        self.assertEqual(results[0]["response_type"], "direct")


if __name__ == "__main__":
    unittest.main()
