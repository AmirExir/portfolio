"""Verify the imported planning preparation and retrieval without external APIs."""

import hashlib
import json
from pathlib import Path
import re
import unittest

from embedding_utils import plan_embedding_reuse
from interview_retrieval import build_candidate, extract_tag_value, rank_candidates
from interview_services import answer_messages, practice_messages
from sync_manager_content import build_chunks, serialize_content


BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "sources" / "transmission_planning_interview_2026-10-19.txt"
SOURCE_SHA256 = "4a453873e0c6b555248c4bd715738a5b789513e2e603b9efbd871305b2896068"
STORY_NUMBERS = {12, 13, 14, 15, 17, 18, 19}


def source_sections(text: str) -> dict[int, dict]:
    """Independently read the supplied numbering, answers, and source caveats."""
    headings = list(re.finditer(r"(?m)^[\s\u2060]*(\d+)\.[\s\u2060]*([^\r\n]+)", text))
    result = {}
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        lines = [line.strip() for line in text[heading.end():end].splitlines() if line.strip()]
        answer = [line for line in lines[1:] if not line.startswith(("Reference:", "Preparation note:"))]
        result[int(heading.group(1))] = {
            "question": heading.group(2).strip(),
            "source_status": lines[0],
            "answer": " ".join(answer),
            "references": [line.removeprefix("Reference:").strip() for line in lines if line.startswith("Reference:")],
            "preparation_note": " ".join(line.removeprefix("Preparation note:").strip() for line in lines if line.startswith("Preparation note:")),
        }
    return result


class PlanningInterviewContentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source_bytes = SOURCE_PATH.read_bytes()
        cls.original = source_sections(cls.source_bytes.decode("utf-8"))
        cls.content = json.loads((BASE_DIR / "manager_content.json").read_text(encoding="utf-8"))
        cls.active_chunks = json.loads((BASE_DIR / "chunks_cleaned.json").read_text(encoding="utf-8"))
        cls.target_chunks, _, _ = build_chunks(cls.active_chunks, cls.content)
        cls.imported = {
            item["source_question"]: item for item in cls.content
            if item["id"].startswith("planning-interview-")
        }
        cls.target_by_id = {chunk.get("source_id"): chunk for chunk in cls.target_chunks}

    def test_source_is_archived_without_byte_changes(self) -> None:
        self.assertEqual(hashlib.sha256(self.source_bytes).hexdigest(), SOURCE_SHA256)
        self.assertEqual(set(self.original), set(range(1, 21)))

    def test_all_twenty_answers_and_caveats_are_preserved(self) -> None:
        self.assertEqual(set(self.imported), set(self.original))
        for number, original in self.original.items():
            with self.subTest(question=number):
                item = self.imported[number]
                self.assertEqual(item["id"], f"planning-interview-{number:02d}")
                self.assertEqual(item["question"], original["question"])
                self.assertEqual(" ".join(item["answer"].split()), " ".join(original["answer"].split()))
                self.assertEqual(item["source_status"], original["source_status"])
                self.assertEqual(item.get("references", []), original["references"])
                self.assertEqual(item.get("preparation_note", ""), original["preparation_note"])
                self.assertEqual(item["source_sha256"], SOURCE_SHA256)
                self.assertEqual((BASE_DIR / item["source_document"]).resolve(), SOURCE_PATH)
                expected_type = "story" if number in STORY_NUMBERS else "direct"
                self.assertEqual(item["response_type"], expected_type)
                if expected_type == "story":
                    for field in ("situation", "task", "action", "result"):
                        self.assertTrue(item[field].strip(), field)
                    self.assertIn("verify", item["source_status"].lower())
                else:
                    self.assertIn("I would", item["answer"])

    def test_metadata_fields_do_not_leak_into_answers_or_star_sections(self) -> None:
        for number, item in self.imported.items():
            with self.subTest(question=number):
                text = self.target_by_id[item["id"]]["text"]
                for tag, field in (
                    ("Question", "question"), ("Answer", "answer"),
                    ("Source Document", "source_document"), ("Source Status", "source_status"),
                    ("Source SHA256", "source_sha256"),
                ):
                    self.assertEqual(extract_tag_value(text, tag), item[field].strip())
                self.assertEqual(extract_tag_value(text, "Source Question"), str(number))
                self.assertEqual(extract_tag_value(text, "References"), "\n".join(item.get("references", [])))
                self.assertEqual(extract_tag_value(text, "Preparation Note"), item.get("preparation_note", ""))
                verification = extract_tag_value(text, "Verification").lower()
                self.assertIn("user-provided", verification)
                self.assertIn("not been independently verified", verification)
                if number in STORY_NUMBERS:
                    for field in ("situation", "task", "action", "result"):
                        self.assertEqual(extract_tag_value(text, field.title()), item[field].strip())
                if item.get("aliases"):
                    self.assertEqual(extract_tag_value(text, "Aliases"), "; ".join(item["aliases"]))

    def test_material_experience_qualifiers_remain_in_evidence(self) -> None:
        for number in (12, 13):
            with self.subTest(question=number):
                self.assertIn("potential", self.imported[number]["result"].lower())
                self.assertIn("$300,000", self.imported[number]["answer"])
        self.assertIn("potential external-fee avoidance, not realized net savings", self.imported[12]["preparation_note"])
        self.assertIn("two projects distinct", self.imported[13]["preparation_note"])
        self.assertIn("The project itself was feasible", self.imported[17]["answer"])
        self.assertIn("colleague’s own achievement", self.imported[19]["preparation_note"])

    def test_every_exact_question_retrieves_its_imported_answer(self) -> None:
        for number, item in self.imported.items():
            with self.subTest(question=number):
                focus = "Project management" if number == 14 else "Transmission planning"
                ranked = rank_candidates(item["question"], self.target_chunks, {}, focus=focus)
                self.assertTrue(ranked)
                self.assertEqual(self.target_chunks[ranked[0]["index"]].get("source_id"), item["id"])

    def test_representative_queries_find_the_new_material(self) -> None:
        queries = {
            1: "Walk us through the ERCOT transmission planning process",
            4: "1,000 MW solar-plus-storage interconnection technical studies",
            5: "750 MW hyperscale data center interconnection energization schedule",
            18: "Tell us about a time you identified a significant technical error or modeling inconsistency",
            20: "ERCOT customer development team conflicting priorities scope cost schedule",
        }
        for number, query in queries.items():
            with self.subTest(question=number):
                ranked = rank_candidates(query, self.target_chunks, {}, focus="Transmission planning")
                ids = [self.target_chunks[candidate["index"]].get("source_id") for candidate in ranked]
                self.assertIn(self.imported[number]["id"], ids)

    def test_exact_story_alias_precedes_a_more_general_planning_story(self) -> None:
        question = "Tell me about a time you resolved a technical planning disagreement."
        self.assertIn(question, self.imported[15]["aliases"])
        for scores in ({}, dict.fromkeys(range(len(self.target_chunks)), 0.5)):
            with self.subTest(semantic_scores=bool(scores)):
                ranked = rank_candidates(question, self.target_chunks, scores, focus="Transmission planning")
                self.assertEqual(self.target_chunks[ranked[0]["index"]].get("source_id"), self.imported[15]["id"])

    def test_ems_example_requires_an_appropriate_role_focus(self) -> None:
        item = self.imported[14]
        chunk = self.target_by_id[item["id"]]
        self.assertEqual(rank_candidates(item["question"], [chunk], {0: 0.99}, focus="Transmission planning"), [])
        for focus in ("All roles", "Project management"):
            with self.subTest(focus=focus):
                ranked = rank_candidates(item["question"], [chunk], {0: 0.99}, focus=focus)
                self.assertEqual([candidate["index"] for candidate in ranked], [0])

    def test_incremental_plan_reuses_all_unchanged_evidence(self) -> None:
        # Simulate the previous published corpus even after this batch is indexed.
        introduced = {item["id"] for item in self.imported.values()}
        introduced.add("story-pec-annual-transmission-assessment")
        previous = [chunk for chunk in self.active_chunks if chunk.get("source_id") not in introduced]
        indices, missing = plan_embedding_reuse(previous, self.target_chunks)
        expected_new = {chunk["text"].strip() for chunk in self.target_chunks if chunk.get("source_id") in introduced}
        self.assertEqual(len(expected_new), 21)
        self.assertEqual(set(missing), expected_new)
        for chunk, index in zip(self.target_chunks, indices):
            if chunk.get("source_id") not in introduced:
                self.assertLess(index, len(previous))
                self.assertEqual(previous[index]["text"].strip(), chunk["text"].strip())
        synchronized, _, _ = build_chunks(self.target_chunks, self.content)
        self.assertEqual(synchronized, self.target_chunks)
        _, missing_after_sync = plan_embedding_reuse(synchronized, self.target_chunks)
        self.assertEqual(missing_after_sync, [])

    def test_invalid_source_metadata_is_rejected(self) -> None:
        item = self.imported[1]
        for field, value in (("source_question", True), ("source_question", 0), ("source_sha256", "not-a-hash"), ("references", "invented citation")):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    serialize_content({**item, field: value})

    def test_answer_and_practice_prompts_preserve_source_limits(self) -> None:
        item = self.imported[4]
        chunk = self.target_by_id[item["id"]]
        candidate = build_candidate(1, 0, chunk)
        answer_prompt = answer_messages(item["question"], candidate, "Desired role")
        practice_prompt = practice_messages(item["question"], "I completed the project.", chunk["text"], "Desired role")
        for messages in (answer_prompt, practice_prompt):
            with self.subTest(prompt=messages[0]["content"][:40]):
                instructions = messages[0]["content"].lower()
                self.assertIn("i would", instructions)
                self.assertIn("not independently verified", instructions)
                self.assertIn("potential fee avoidance", instructions)
                self.assertIn("tool named only in the question", instructions)
                self.assertIn("effective", instructions)
                self.assertIn("preparation notes", instructions)
                self.assertNotIn(chunk["text"], messages[0]["content"])
        self.assertIn(chunk["text"], answer_prompt[1]["content"])
        self.assertEqual(json.loads(practice_prompt[1]["content"])["selected_context"], chunk["text"])


if __name__ == "__main__":
    unittest.main()
