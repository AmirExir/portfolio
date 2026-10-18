"""Offline regressions; mocked simulator success is not licensed validation."""
import json
from pathlib import Path
import tempfile
import unittest

from psse_preflight import python_blocks, review_code
from psse_validate_case import validate_case


class PreflightTests(unittest.TestCase):
    def test_parse_and_never_execute(self):
        report = review_code("raise RuntimeError('must not execute')", psse_version="35.6.0")
        self.assertTrue(report["syntax_valid"])
        self.assertFalse(review_code("if:", psse_version="35.6.0")["syntax_valid"])

    def test_aliases_and_overwrite_are_reported(self):
        code = 'import psspy as p\np.case("C:/input.sav")\np.save("c:\\\\input.sav")\n'
        report = review_code(code, psse_version="35.6.0")
        codes = {item["code"] for item in report["findings"]}
        self.assertTrue({"initialization", "return_code", "source_overwrite"} <= codes)
        self.assertTrue(report["has_errors"])

    def test_signature_binding_and_version_are_explicit(self):
        catalog = {"schema_version": 1, "psse_version": "test-version", "source": "synthetic fixture",
                   "functions": {"case": [{"name": "path", "kind": "POSITIONAL_OR_KEYWORD", "required": True}]}}
        report = review_code("psspy.case()", psse_version="test-version", catalog=catalog)
        self.assertTrue(any(item["code"] == "signature" for item in report["findings"]))
        self.assertEqual(review_code('psspy.case("copy.sav")', psse_version="test-version", catalog=catalog)["calls"][0]["signature_status"], "call shape matches supplied catalog")
        with self.assertRaisesRegex(ValueError, "version"):
            review_code("psspy.case()", psse_version="another", catalog=catalog)

    def test_dynamic_save_and_assigned_ierr_remain_review_items(self):
        report = review_code("ierr = psspy.psseinit(100)\nierr = psspy.save(output)", psse_version="35.6.0")
        self.assertIn("save_path_unknown", {item["code"] for item in report["findings"]})
        self.assertTrue(any(item["code"] == "return_code" for item in report["findings"]))
        self.assertIn("unverified", report["status"])

    def test_fences(self):
        self.assertEqual(python_blocks('Example:\n```python\nx=1\n```'), ['x=1\n'])


class FakePSSE:
    def __init__(self, failure=0, convergence=0):
        self.paths = []
        self.failure, self.convergence = failure, convergence
    def psseinit(self, _buses): return 0
    def psseversion(self): return ("PSS/E", 35, 6, 0, "fixture", "fixture")
    def case(self, path): self.paths.append(path); return 0
    def fnsl(self, _options): return self.failure
    def solved(self): return self.convergence
    def save(self, path): Path(path).write_bytes(b"derived synthetic case"); return 0


class CaseValidationTests(unittest.TestCase):
    def run_case(self, api, root):
        source = root / "original.sav"
        source.write_bytes(b"synthetic test bytes, not a real case")
        report = validate_case(api, source, root / "new-run", version="35.6.0", buses=10,
                               solve_options=[0] * 8, assumptions="synthetic wrapper test only")
        self.assertEqual(source.read_bytes(), b"synthetic test bytes, not a real case")
        self.assertTrue(all(Path(path).name == "input_copy.sav" for path in api.paths))
        self.assertEqual(json.loads((root / "new-run" / "validation.json").read_text())["status"], report["status"])
        return report

    def test_copy_solve_restore_and_refuse_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = self.run_case(FakePSSE(), root)
            self.assertEqual(report["status"], "solver smoke test completed")
            self.assertEqual(report["engineering_validation"], "not performed")
            with self.assertRaises(FileExistsError):
                self.run_case(FakePSSE(), root)

    def test_nonzero_return_or_unconverged_never_saves_success(self):
        for api in (FakePSSE(failure=1), FakePSSE(convergence=2)):
            with tempfile.TemporaryDirectory() as directory:
                report = self.run_case(api, Path(directory))
                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["restored_input_copy"])
                self.assertFalse((Path(directory) / "new-run" / "solved_copy.sav").exists())

    def test_unreadable_source_still_writes_failure_and_restore_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.sav"
            source.write_bytes(b"synthetic fixture")
            api = FakePSSE()
            def external_removal(_options):
                source.unlink()  # Simulate another process removing the original.
                return 0
            api.fnsl = external_removal
            report = validate_case(api, source, root / "run", version="35.6.0", buses=10,
                                   solve_options=[0] * 8, assumptions="synthetic fixture")
            self.assertEqual(report["status"], "failed")
            self.assertIsNone(report["source_unchanged"])
            self.assertTrue(report["restored_input_copy"])
            saved = json.loads((root / "run" / "validation.json").read_text())
            self.assertIn("FileNotFoundError", saved["source_integrity_error"])


class PreflightUITests(unittest.TestCase):
    def test_review_and_report_survive_rerun(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('from psse_preflight_ui import render_preflight\nrender_preflight("```python\\npsspy.save(output)\\n```")').run()
        app.text_input[0].set_value("35.6.0")
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertEqual(len(app.error), 0)
        self.assertTrue(app.session_state["psse_preflight_reports"])
        app.run()
        self.assertEqual(len(app.dataframe), 1)

    def test_invalid_version_clears_previous_report(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_string('from psse_preflight_ui import render_preflight\nrender_preflight("psspy.fnsl()")').run()
        app.text_input[0].set_value("35.6.0")
        app.button[0].click().run()
        app.text_input[0].set_value("")
        app.button[0].click().run()
        self.assertEqual(len(app.exception), 0, str(app.exception))
        self.assertEqual(len(app.error), 1)
        self.assertEqual(len(app.dataframe), 0)


if __name__ == "__main__":
    unittest.main()
