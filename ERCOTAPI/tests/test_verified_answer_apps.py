"""Exercise the real app submission branches without starting Streamlit or APIs.

The harness executes each application's actual function definitions and top-level
chat submission branch. Only startup/UI infrastructure, retrieval, generation,
and the verifier are replaced. This tests the boundary that must prevent a raw
draft from reaching either rendering or conversation history.
"""

from __future__ import annotations

import ast
import copy
import re
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from DWG_SSWG_Chatbot.chat_completion_handling import assess_chat_completion
from ERCOTAPI.rag_ingestion import retrieval
from ERCOTAPI.rag_ingestion.response_handling import (
    ANSWER_VERIFICATION_POLICY,
    assess_response,
    compact_chat_messages,
    retain_verified_history,
)


ROOT = Path(__file__).resolve().parents[2]
APPS = (
    "chatbot_ercot_all_in_one/ercot_assistant_app.py",
    "chatbot_ercot/ercotAIassistant.py",
    "chatbot_ercot_nodalprotocols/ercotProtocolsAssistant.py",
    "chatbot_ERCOT_RI/ercotAIassistant.py",
    "DWG_SSWG_Chatbot/ERCOT_DWG_SSWG_AI_Assistant.py",
)
QUESTION = "Explain the generator model submission requirement."
RAW_DRAFT = "UNSUPPORTED_RAW_CLAIM [E1].\nSupported source statement [E1]."
VERIFIED_TEXT = "Retained supported source statement [E1]."
ABSTENTION = "The available documents do not establish this conclusively."


class _Stopped(Exception):
    """Streamlit stops executing the current script."""


class _Streamlit:
    def __init__(self, events):
        self.events = events
        self.session_state = SimpleNamespace(messages=[])
        self.output: list[str] = []
        self.assistant_output: list[str] = []

    def chat_input(self, *_args, **_kwargs):
        return QUESTION

    def chat_message(self, role):
        def markdown(value, **_kwargs):
            self.output.append(str(value))
            if role == "assistant":
                self.events.append("render")
                self.assistant_output.append(str(value))

        return SimpleNamespace(markdown=markdown)

    def cache_resource(self, function=None, **_kwargs):
        return function if function is not None else lambda wrapped: wrapped

    def spinner(self, *_args, **_kwargs):
        return nullcontext()

    expander = spinner

    def write(self, value, **_kwargs):
        self.output.append(str(value))

    error = write
    warning = write
    caption = write
    info = write

    def stop(self):
        raise _Stopped()


class VerifiedAnswerAppTests(unittest.TestCase):
    def run_app(self, relative, *, retry=False, verifier_mode="partial", incomplete=False, old_history=False):
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        # Select the outer submission branch, including planning's deterministic
        # document-visibility branches, without executing index or UI startup.
        submission = [
            node for node in tree.body
            if isinstance(node, ast.If)
            and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "chat_message"
                and child.args
                and isinstance(child.args[0], ast.Constant)
                and child.args[0].value == "user"
                for child in ast.walk(node)
            )
        ]
        self.assertEqual(len(submission), 1, relative)
        events = []
        st = _Streamlit(events)
        if old_history:
            st.session_state.messages = [
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "UNSUPPORTED_RAW_CLAIM from an old session"},
                {"role": "assistant", "content": "UNSUPPORTED_RAW_CLAIM from an old policy",
                 "verification_policy": "claim-verification-v0"},
                {"role": "assistant", "content": "Previously checked statement [E1].",
                 "verification_policy": ANSWER_VERIFICATION_POLICY},
            ]
        generation_calls = []
        verification_calls = []
        footer_calls = []
        # Long unique passages force genuine word-budget truncation, including
        # a partly visible final passage. This catches checking hidden evidence.
        available = [
            {
                "chunk_id": f"chunk-{number}",
                "document_id": f"document-{number}",
                "chunk_index": 0,
                "text": " ".join([f"sourceword{number}"] * 3100),
                "source_path": f"sources/document-{number}.txt",
                "source_kind": "Planning Guide",
                "source_authority": "ERCOT",
                "evidence_id": f"E{number}",
                "as_of": "2026-11-10",
            }
            for number in range(1, 17)
        ]

        def retrieve_chunks(_question, _index, *, top_k, **_kwargs):
            return copy.deepcopy(available[:top_k])

        def retrieve_bundle(_question, _index, *, top_k, **_kwargs):
            return {
                "chunks": retrieve_chunks(_question, _index, top_k=top_k),
                "analysis": {"as_of": "2026-11-10", "intent": "current_requirement"},
                "answer_contract": "Cite supporting evidence IDs.",
                "change_reports": [],
            }

        def generate(**kwargs):
            events.append("generate")
            generation_calls.append(kwargs)
            is_incomplete = incomplete or (retry and len(generation_calls) == 1)
            return SimpleNamespace(
                status="incomplete" if is_incomplete else "completed",
                output_text=RAW_DRAFT,
                incomplete_details=SimpleNamespace(reason="max_output_tokens")
                if is_incomplete else None,
                choices=[SimpleNamespace(
                    finish_reason="length" if is_incomplete else "stop",
                    message=SimpleNamespace(content=RAW_DRAFT, refusal=None),
                )],
            )

        client = SimpleNamespace(
            responses=SimpleNamespace(create=generate),
            chat=SimpleNamespace(completions=SimpleNamespace(create=generate)),
        )

        def verify(answer, chunks, **kwargs):
            events.append("verify")
            verification_calls.append((answer, copy.deepcopy(chunks), kwargs))
            if verifier_mode == "exception":
                raise RuntimeError("Verifier service unavailable")
            if verifier_mode == "empty":
                text, cited = "", []
            elif verifier_mode == "abstain":
                text, cited = ABSTENTION, []
            else:
                text, cited = VERIFIED_TEXT, list(chunks[:1])
            return SimpleNamespace(
                text=text,
                cited_chunks=cited,
                status="partial" if cited else "abstained",
                notice="Source verification completed; unsupported claims were withheld.",
                counts={"supported": len(cited), "unsupported": 1},
                checks=[],
            )

        def footer(chunks, **kwargs):
            events.append("footer")
            footer_calls.append(copy.deepcopy(chunks))
            return retrieval.format_source_list(chunks, **kwargs)

        namespace = {
            **vars(retrieval),
            "__file__": str(path),
            "__name__": "app_submission_test",
            "st": st,
            "re": re,
            "Path": Path,
            "rag_index": SimpleNamespace(chunks=available),
            "chunks": available,
            "sources": [],
            "valid_functions": set(),
            "prompt": QUESTION,
            "assess_response": assess_response,
            "assess_chat_completion": assess_chat_completion,
            "compact_chat_messages": compact_chat_messages,
            "ANSWER_VERIFICATION_POLICY": ANSWER_VERIFICATION_POLICY,
            "retain_verified_history": retain_verified_history,
            "verify_answer": verify,
        }
        definitions = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
            + [node for node in tree.body if isinstance(node, ast.FunctionDef)],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(definitions), str(path), "exec"), namespace)
        namespace.update({
            "get_openai_client": lambda: client,
            "safe_openai_call": lambda function, **kwargs: function(**kwargs),
            "retrieve_chunks": retrieve_chunks,
            "retrieve_requirement_evidence": retrieve_bundle,
            "format_source_list": footer,
        })
        if old_history:
            replay = [
                node for node in tree.body
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "retain_verified_history"
                ) or (
                    isinstance(node, ast.For)
                    and any(isinstance(child, ast.Attribute) and child.attr == "chat_message"
                            for child in ast.walk(node))
                )
            ]
            self.assertEqual(len(replay), 2, relative)
            exec(compile(ast.Module(body=replay, type_ignores=[]), str(path), "exec"), namespace)
        initial_assistant_count = len(st.assistant_output)
        try:
            exec(compile(ast.Module(body=submission, type_ignores=[]), str(path), "exec"), namespace)
        except _Stopped:
            pass
        stored = [message["content"] for message in st.session_state.messages if message["role"] == "assistant"]
        self.assertEqual(st.assistant_output, stored, relative)
        for message in st.session_state.messages:
            if message["role"] == "assistant":
                self.assertEqual(message.get("verification_policy"), ANSWER_VERIFICATION_POLICY, relative)
        self.assertNotIn("UNSUPPORTED_RAW_CLAIM", "\n".join(st.output + stored), relative)
        if verification_calls:
            answer, checked, kwargs = verification_calls[0]
            self.assertEqual(answer, RAW_DRAFT)
            self.assertIs(kwargs["client"], client)
            self.assertEqual(kwargs["question"], QUESTION)
            self.assertEqual(kwargs["as_of"], "2026-11-10")
            generation = generation_calls[-1]
            messages = generation.get("input", generation.get("messages"))
            system_text = messages[0]["content"]
            visible_ids = re.findall(r"Evidence \[(E\d+)\]", system_text)
            self.assertEqual([chunk["evidence_id"] for chunk in checked], visible_ids, relative)
            for chunk in checked:
                self.assertIn(chunk["text"], system_text, relative)
            self.assertTrue(any(chunk.get("context_truncated") for chunk in checked), relative)
        if footer_calls:
            self.assertLess(events.index("verify"), events.index("footer"), relative)
        if stored:
            self.assertLess(events.index("verify"), len(events) - 1 - events[::-1].index("render"), relative)
        for generation in generation_calls:
            self.assertNotIn("UNSUPPORTED_RAW_CLAIM", str(generation), relative)
        return SimpleNamespace(
            stored=stored[initial_assistant_count:], events=events, footer_calls=footer_calls,
            generation_calls=generation_calls, verification_calls=verification_calls,
            replayed=st.assistant_output[:initial_assistant_count],
        )

    def test_all_five_apps_render_and_store_only_verified_claims(self):
        for app in APPS:
            with self.subTest(app=app):
                run = self.run_app(app)
                self.assertEqual(len(run.verification_calls), 1)
                self.assertEqual(len(run.stored), 1)
                self.assertIn(VERIFIED_TEXT, run.stored[0])
                self.assertEqual([chunk["evidence_id"] for chunk in run.footer_calls[0]], ["E1"])

    def test_retry_verifies_only_evidence_visible_to_successful_generation(self):
        for app in APPS:
            with self.subTest(app=app):
                run = self.run_app(app, retry=True)
                self.assertEqual(len(run.generation_calls), 2)
                self.assertEqual(len(run.verification_calls), 1)
                self.assertIn(VERIFIED_TEXT, run.stored[0])
                checked = run.verification_calls[0][1]
                self.assertLess(len(checked), 5 if app in APPS[1:3] else 14)

    def test_abstention_never_falls_back_to_retrieved_source_footer(self):
        for app in APPS:
            with self.subTest(app=app):
                run = self.run_app(app, verifier_mode="abstain")
                self.assertIn(ABSTENTION, run.stored[0])
                self.assertTrue(all(not chunks for chunks in run.footer_calls))
                self.assertNotIn("Retrieved sources", run.stored[0])

    def test_verifier_failure_or_empty_result_never_exposes_draft(self):
        for app in APPS:
            for mode in ("exception", "empty"):
                with self.subTest(app=app, mode=mode):
                    run = self.run_app(app, verifier_mode=mode)
                    self.assertEqual(len(run.verification_calls), 1)
                    self.assertEqual(run.stored, [])
                    self.assertEqual(run.footer_calls, [])

    def test_incomplete_generation_never_reaches_verification_or_output(self):
        for app in APPS:
            with self.subTest(app=app):
                run = self.run_app(app, incomplete=True)
                self.assertEqual(len(run.generation_calls), 2)
                self.assertEqual(run.verification_calls, [])
                self.assertEqual(run.footer_calls, [])
                self.assertEqual(run.stored, [])

    def test_each_app_imports_the_shared_verifier(self):
        for app in APPS:
            with self.subTest(app=app):
                tree = ast.parse((ROOT / app).read_text(encoding="utf-8"))
                imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
                self.assertTrue(any(
                    node.module == "ERCOTAPI.rag_ingestion.claim_verification"
                    and any(alias.name == "verify_answer" for alias in node.names)
                    for node in imports
                ))

    def test_old_unverified_history_is_removed_before_replay_and_generation(self):
        for app in APPS:
            with self.subTest(app=app):
                run = self.run_app(app, old_history=True)
                self.assertEqual(run.replayed, ["Previously checked statement [E1]."])
                self.assertEqual(len(run.stored), 1)
                self.assertIn(VERIFIED_TEXT, run.stored[0])


if __name__ == "__main__":
    unittest.main()
