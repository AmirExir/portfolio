from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import psse_rag_context as context


ROOT = Path(__file__).resolve().parents[1]


def load_function(relative_path, name, namespace):
    """Exercise an actual UI helper without running Streamlit or API clients."""
    source = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    node = next(item for item in source.body if isinstance(item, ast.FunctionDef)
                and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), relative_path, "exec"), namespace)
    return namespace[name]


class HybridContextTests(unittest.TestCase):
    def setUp(self):
        self.docs = [{"id": f"doc{i}", "text": f"documentation {i}"} for i in range(10)]
        self.code = [{"id": f"code{i}", "text": f"code {i}"} for i in range(4)]

    @patch.object(context, "search_code")
    def test_code_and_documentation_both_survive_top_k(self, search):
        search.return_value = self.code
        result = context.hybrid_chunks("dynamic retries", self.docs)
        self.assertEqual(len(result), 10)
        self.assertEqual([c["id"] for c in result[:4]], ["code0", "doc0", "code1", "doc1"])
        self.assertEqual(sum(c["id"].startswith("code") for c in result), 4)

    @patch.object(context, "search_code", return_value=[])
    def test_missing_or_unmatched_code_retains_documentation(self, _search):
        self.assertEqual(context.hybrid_chunks("query", self.docs), self.docs)

    @patch.object(context, "search_code", side_effect=ValueError("index corrupt"))
    def test_failed_code_search_is_not_silently_ignored(self, _search):
        with self.assertRaisesRegex(ValueError, "index corrupt"):
            context.hybrid_chunks("query", self.docs)

    @patch.object(context, "search_code")
    def test_nonpositive_budget_makes_no_query(self, search):
        self.assertEqual(context.hybrid_chunks("query", self.docs, k=0), [])
        search.assert_not_called()

    @patch.object(context, "search_code")
    def test_duplicate_ids_are_not_repeated(self, search):
        search.return_value = [self.docs[0], *self.code]
        result = context.hybrid_chunks("query", self.docs, k=3)
        self.assertEqual(len({c["id"] for c in result}), 3)

    @patch.object(context, "index_status", return_value={"available": False})
    def test_missing_index_is_visible_to_user(self, _status):
        self.assertIn("not installed", context.code_index_caption())

    def test_reference_labels_include_provenance(self):
        labels = context.reference_labels([{
            "source_path": "aelab25_app/dynamic_batch_executor.py",
            "start_line": 10, "end_line": 50,
            "revision": "abcdef123456789", "file_sha256": "123456abcdef789",
        }])
        self.assertIn("dynamic_batch_executor.py:10-50", labels[0])
        self.assertIn("abcdef123456", labels[0])
        self.assertIn("123456abcdef", labels[0])

    def test_example_functions_are_not_presented_as_documented_apis(self):
        code = {"source_path": "script.py", "text": "psspy.fnsl(); psspy.example_only()"}
        self.assertEqual(context.example_only_functions(
            "psspy.fnsl(); psspy.example_only(); psspy.invented()",
            {"fnsl"}, [code],
        ), ["example_only"])

    @patch.object(context, "search_code")
    def test_multiagent_retrieval_uses_code_before_token_limit(self, search):
        search.return_value = self.code
        seen = []
        def limit(chunks, max_tokens):
            seen.extend(chunks)
            return chunks[:2]
        find = load_function("chatbot_PSSEBOT_MultiAgent/retriever.py", "find_relevant_chunks", {
            "find_top_k_chunks": lambda *args, **kwargs: self.docs,
            "hybrid_chunks": context.hybrid_chunks,
            "limit_chunks_by_token_budget": limit,
        })
        result = find("retry after crash", [], [], max_tokens=100)
        self.assertEqual([c["id"] for c in result], ["code0", "doc0"])
        self.assertEqual(len(seen), 10)

    @patch.object(context, "search_code")
    def test_standard_retrieval_merges_documentation_and_code(self, search):
        search.return_value = self.code
        class Array:
            def reshape(self, *_args): return self
            def flatten(self): return self
            def argsort(self): return list(range(10))
        find = load_function("chatbot_PSSEBOT/psse_bot.py", "find_top_k_matches", {
            "embed_query": lambda _query: [1.0],
            "np": type("NP", (), {"array": lambda *args: Array()}),
            "cosine_similarity": lambda *args: Array(),
            "hybrid_chunks": context.hybrid_chunks,
        })
        result = find("dynamic retries", self.docs, object(), k=3)
        self.assertEqual([c["id"] for c in result], ["code0", "doc9", "code1"])

    def test_all_prompts_explain_repository_evidence_limits(self):
        build = load_function("chatbot_PSSEBOT/psse_bot.py", "build_system_prompt", {
            "CODE_REFERENCE_RULES": context.CODE_REFERENCE_RULES,
        })
        self.assertIn(context.CODE_REFERENCE_RULES, build("reference")["content"])
        for filename in ("planner.py", "executor.py"):
            # Compile and invoke the real function with a fake completion service.
            calls = []
            def request(_create, primary, **_kwargs):
                calls.append(primary)
                return type("Answer", (), {"usable": True, "text": "1. Review source"})()
            namespace = {
                "CODE_REFERENCE_RULES": context.CODE_REFERENCE_RULES,
                "count_tokens": lambda _text: 1,
                "client": type("Client", (), {"responses": type("Responses", (), {"create": None})()})(),
                "request_visible_answer": request,
                "re": __import__("re"),
            }
            name = "plan_tasks" if filename == "planner.py" else "run_executor"
            fn = load_function(f"chatbot_PSSEBOT_MultiAgent/{filename}", name, namespace)
            if name == "plan_tasks":
                fn("query", [{"text": "reference"}])
            else:
                fn("query", "reference", set())
            self.assertIn(context.CODE_REFERENCE_RULES, calls[0]["input"][0]["content"])


if __name__ == "__main__":
    unittest.main()
