"""Local code-index regressions using temporary synthetic Git repositories.

These tests never import indexed source, invoke PSS/E, use utility study data,
or make embedding/API requests.
"""

from __future__ import annotations

from contextlib import closing
import hashlib
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from unittest import mock

import psse_code_index as code_index


class PsseCodeIndexTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-psse-index-")
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.source = self.folder / "source"
        self.source.mkdir()
        self.index = self.folder / "cache" / "code.sqlite3"
        self.git("init", "--quiet")

    def git(self, *arguments):
        return subprocess.run(
            ["git", "-C", str(self.source), *arguments], check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ).stdout.strip()

    def write(self, relative, text):
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def commit(self, message="Synthetic fixture"):
        self.git("add", "--all")
        self.git("-c", "user.name=Synthetic Test", "-c", "user.email=test@example.invalid",
                 "commit", "--quiet", "--allow-empty", "-m", message)
        return self.git("rev-parse", "HEAD")

    def refresh(self):
        return code_index.refresh_index(self.source, self.index)

    def search(self, query, k=20):
        return code_index.search_code(query, self.index, k=k)

    def test_unchanged_refresh_preserves_chunk_ids_and_does_not_duplicate_hits(self):
        self.write("aelab25_app/synthetic.py", "def persistent_anchor():\n    return 'persistent_anchor'\n")
        self.commit()
        self.refresh()
        initial = self.search("persistent_anchor")
        self.assertTrue(initial)

        self.refresh()
        refreshed = self.search("persistent_anchor")

        self.assertEqual({hit["id"] for hit in initial}, {hit["id"] for hit in refreshed})
        self.assertEqual(len(refreshed), len({hit["id"] for hit in refreshed}))
        self.assertEqual(initial, refreshed)
        self.assertTrue(code_index.index_status(self.index)["available"])

    def test_changed_file_replaces_old_text_and_preserves_unrelated_chunks(self):
        self.write("aelab25_app/changing.py", "def old_refresh_marker():\n    return 1\n")
        self.write("aelab25_app/stable.py", "def stable_refresh_marker():\n    return 2\n")
        self.commit()
        self.refresh()
        stable_before = {hit["id"] for hit in self.search("stable_refresh_marker")}
        self.assertTrue(self.search("old_refresh_marker"))

        self.write("aelab25_app/changing.py", "def new_refresh_marker():\n    return 3\n")
        self.commit("Changed synthetic helper")
        self.refresh()

        self.assertEqual(self.search("old_refresh_marker"), [])
        self.assertTrue(self.search("new_refresh_marker"))
        self.assertEqual(stable_before, {hit["id"] for hit in self.search("stable_refresh_marker")})

    def test_deleted_source_is_removed_while_retained_source_remains_searchable(self):
        deleted = self.write("aelab25_app/deleted.py", "def removed_source_marker():\n    return 1\n")
        self.write("aelab25_app/retained.py", "def retained_source_marker():\n    return 2\n")
        self.commit()
        self.refresh()
        self.assertTrue(self.search("removed_source_marker"))

        deleted.unlink()
        self.commit("Removed synthetic helper")
        report = self.refresh()

        self.assertEqual(self.search("removed_source_marker"), [])
        self.assertTrue(self.search("retained_source_marker"))
        self.assertEqual(report["deleted"], ["aelab25_app/deleted.py"])
        self.assertEqual(report["deleted_count"], 1)
        self.assertEqual(report["files"], 1)

    def test_source_metadata_identifies_exact_file_hash_and_line_slice(self):
        text = (
            '"""Synthetic code fixture."""\n\n'
            "def start_dynamic_case(psspy):\n"
            '    """Initialize the synthetic run."""\n'
            '    ierr = psspy.strt(0, "synthetic.out")\n'
            "    if ierr:\n"
            '        raise RuntimeError("Synthetic startup failed")\n'
            "    return ierr\n"
        )
        path = self.write("aelab25_app/dynamics.py", text)
        revision = self.commit()
        self.refresh()
        hits = self.search("psspy.strt")
        self.assertTrue(hits)

        for hit in hits:
            self.assertEqual(hit["source_path"], "aelab25_app/dynamics.py")
            self.assertEqual(hit["file_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(hit["revision"], revision)
            self.assertIsInstance(hit["start_line"], int)
            self.assertIsInstance(hit["end_line"], int)
            self.assertGreaterEqual(hit["start_line"], 1)
            self.assertGreaterEqual(hit["end_line"], hit["start_line"])
            self.assertLessEqual(hit["end_line"], len(text.splitlines()))
            expected = "\n".join(text.splitlines()[hit["start_line"]-1:hit["end_line"]])
            self.assertEqual(hit["source_text"].rstrip("\n"), expected)
            self.assertIn(expected, hit["text"])
            self.assertIn(f"{hit['source_path']}:{hit['start_line']}-{hit['end_line']}", hit["text"])
            self.assertIn(revision, hit["text"])
            self.assertIn(hit["file_sha256"], hit["text"])
            self.assertIn("not authoritative", hit["text"])

    def test_same_basename_files_keep_distinct_source_identity(self):
        for folder in ("aelab25_app/first", "aelab25_app/second"):
            self.write(folder + "/helper.py", "def shared_exact_symbol():\n    return 1\n")
        self.commit()
        self.refresh()
        hits = self.search("shared_exact_symbol")
        self.assertEqual({hit["source_path"] for hit in hits},
                         {"aelab25_app/first/helper.py", "aelab25_app/second/helper.py"})
        self.assertEqual(len(hits), len({hit["id"] for hit in hits}))

    def test_symbol_and_api_search_are_bounded_and_punctuation_is_safe(self):
        self.write("aelab25_app/startup.py", "def start_dynamic_case(psspy):\n    return psspy.strt(0, 'synthetic.out')\n")
        self.write("aelab25_app/other.py", "def unrelated_helper():\n    return 'other topic'\n")
        self.commit()
        self.refresh()

        for query in ("start_dynamic_case", "psspy.strt", "psspy.strt(0, 'synthetic.out');"):
            with self.subTest(query=query):
                hits = self.search(query, k=1)
                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0]["source_path"], "aelab25_app/startup.py")
        for query in ("", "   ", '" : () {} [] ! --', "zzzz_no_such_symbol_987654321"):
            with self.subTest(query=query):
                self.assertEqual(self.search(query), [])

    def test_source_is_read_without_executing_top_level_python(self):
        sentinel = self.folder / "must_not_be_created.txt"
        self.write("aelab25_app/no_execution.py",
                   "from pathlib import Path\n"
                   f"Path({str(sentinel)!r}).write_text('Source was executed')\n"
                   "raise RuntimeError('Indexed source must never run')\n\n"
                   "def noexecute_probe():\n    return 1\n")
        self.commit()
        self.refresh()

        self.assertFalse(sentinel.exists())
        self.assertTrue(self.search("noexecute_probe"))

    def test_symlink_escape_is_not_indexed(self):
        outside = self.folder / "outside.py"
        outside.write_text("def symlink_escape_marker():\n    return 1\n", encoding="utf-8")
        self.write("aelab25_app/inside.py", "def legitimate_inside_marker():\n    return 2\n")
        link = self.source / "aelab25_app/linked.py"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(f"Temporary symlink fixture unavailable: {exc}")
        self.commit()
        self.refresh()

        self.assertEqual(self.search("symlink_escape_marker"), [])
        self.assertTrue(self.search("legitimate_inside_marker"))

    def test_missing_index_reports_unavailable_without_creating_database(self):
        status = code_index.index_status(self.index)
        self.assertFalse(status["available"])
        self.assertFalse(self.index.exists())

    def test_corrupt_index_reports_error_without_overwriting_original_bytes(self):
        self.index.parent.mkdir()
        original = b"synthetic-corrupt-database-not-sqlite"
        self.index.write_bytes(original)

        with self.assertRaises(ValueError):
            code_index.index_status(self.index)
        with self.assertRaises(ValueError):
            self.search("psspy.strt")
        self.assertEqual(self.index.read_bytes(), original)

    def test_default_index_path_honors_explicit_environment_override(self):
        target = self.folder / "custom" / "code.sqlite3"
        with mock.patch.dict(os.environ, {"PSSE_CODE_INDEX": str(target)}):
            self.assertEqual(Path(code_index.default_index_path()), target)
        self.assertFalse(target.exists())

    def test_refresh_reports_identical_files_and_refuses_empty_source_snapshot(self):
        path = self.write("aelab25_app/incremental.py", "def incremental_marker():\n    return 1\n")
        self.commit()
        initial = self.refresh()
        self.assertEqual(initial["changed"], ["aelab25_app/incremental.py"])
        self.assertEqual(initial["changed_count"], 1)
        self.assertEqual(initial["files"], 1)
        self.assertGreater(initial["chunks"], 0)
        unchanged = self.refresh()
        self.assertEqual(unchanged["changed"], [])
        self.assertEqual(unchanged["unchanged"], ["aelab25_app/incremental.py"])
        self.assertEqual(unchanged["source_hash"], initial["source_hash"])
        self.assertEqual(unchanged["chunks"], initial["chunks"])

        path.unlink()
        self.commit("Remove only indexed source")
        with self.assertRaisesRegex(ValueError, "No eligible|empty"):
            self.refresh()
        preserved = code_index.index_status(self.index)
        self.assertEqual(preserved["files"], initial["files"])
        self.assertEqual(preserved["chunks"], initial["chunks"])
        self.assertEqual(preserved["source_hash"], initial["source_hash"])
        self.assertEqual(preserved["revision"], initial["revision"])

    def test_new_commit_updates_revision_without_replacing_unchanged_chunk_ids(self):
        self.write("aelab25_app/revision.py", "def revision_marker():\n    return 1\n")
        first_revision = self.commit()
        self.refresh()
        before = self.search("revision_marker")
        self.assertEqual({hit["revision"] for hit in before}, {first_revision})

        next_revision = self.commit("New revision with unchanged code")
        report = self.refresh()
        after = self.search("revision_marker")

        self.assertNotEqual(first_revision, next_revision)
        self.assertEqual(report["changed"], [])
        self.assertEqual(report["commit"], next_revision)
        self.assertFalse(report["dirty"])
        self.assertEqual({hit["id"] for hit in after}, {hit["id"] for hit in before})
        self.assertEqual({hit["revision"] for hit in after}, {next_revision})
        self.assertTrue(all(next_revision in hit["text"] and first_revision not in hit["text"] for hit in after))

    def test_dirty_source_has_current_hash_and_explicit_dirty_revision(self):
        path = self.write("aelab25_app/dirty.py", "def committed_marker():\n    return 1\n")
        commit = self.commit()
        self.refresh()
        committed_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.write("aelab25_app/dirty.py", "def working_tree_marker():\n    return 2\n")

        report = self.refresh()
        hits = self.search("working_tree_marker")
        self.assertTrue(report["dirty"])
        self.assertEqual(report["revision"], commit + "-dirty")
        self.assertEqual(self.search("committed_marker"), [])
        self.assertTrue(hits)
        expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertNotEqual(expected_hash, committed_hash)
        self.assertEqual({hit["file_sha256"] for hit in hits}, {expected_hash})
        self.assertEqual({hit["revision"] for hit in hits}, {commit + "-dirty"})

    def test_allowed_code_and_engineering_docs_are_searchable(self):
        allowed = {
            "synthetic_psse_helper.py": "def allowed_root_marker():\n    return 1\n",
            "aelab25_app/helper.py": "def allowed_app_marker():\n    return 1\n",
            "TRABAJO/Python/helper.py": "def allowed_trabajo_marker():\n    return 1\n",
            "DWG_Contingency_Tool_V3-1/helper.py": "def allowed_dwg_marker():\n    return 1\n",
            "Docs/dynamic_simulation.md": "# Synthetic engineering guide\n\nallowed_docs_marker\n",
        }
        for path, content in allowed.items():
            self.write(path, content)
        self.commit()
        self.refresh()

        for marker in ("allowed_root_marker", "allowed_app_marker", "allowed_trabajo_marker", "allowed_dwg_marker", "allowed_docs_marker"):
            with self.subTest(marker=marker):
                self.assertTrue(self.search(marker))

    def test_secrets_study_outputs_tests_and_vendor_sources_are_excluded(self):
        excluded = [
            "vendor/tool.py", "third_party/tool.py", "tests/test_fixture.py", ".venv/lib/helper.py",
            "EVENTs/event.py", "OUTs/output.py", "LOGs/log.py", "LIBs/library.py",
            "TRABAJO/Python/xlrd/helper.py", "TRABAJO/Python/xlwt/helper.py",
            "TRABAJO/Python/curses/helper.py", "TRABAJO/Python/psutil/helper.py",
            "pyPowerGEM_v2601.3/private.py", "dmviwe/helper.py",
            ".streamlit/secrets.toml", ".env", "credentials.py", "AGENTS.md",
            "CASEs/synthetic.sav", "CASEs/synthetic.snp", "EVENTs/synthetic.idv", "EVENTs/synthetic.con",
            "OUTs/synthetic.csv", "Docs/synthetic.xlsx",
        ]
        self.write("aelab25_app/allowed.py", "def public_source_marker():\n    return 1\n")
        for index, path in enumerate(excluded):
            self.write(path, f"excluded_private_marker_{index:03d}\n")
        self.commit()
        report = self.refresh()
        self.assertTrue(self.search("public_source_marker"))
        self.assertEqual(report["files"], 1)
        for index, path in enumerate(excluded):
            with self.subTest(path=path):
                self.assertEqual(self.search(f"excluded_private_marker_{index:03d}"), [])

    def test_untracked_source_is_not_silently_added_to_git_scoped_index(self):
        self.write("aelab25_app/tracked.py", "def tracked_source_marker():\n    return 1\n")
        revision = self.commit()
        self.write("aelab25_app/untracked_private.py", "def untracked_private_marker():\n    return 2\n")
        report = self.refresh()

        self.assertTrue(self.search("tracked_source_marker"))
        self.assertEqual(self.search("untracked_private_marker"), [])
        self.assertEqual(report["revision"], revision)

    def test_fatal_mid_refresh_failure_rolls_back_changed_files_and_metadata(self):
        self.write("aelab25_app/first.py", "def rollback_old_first():\n    return 1\n")
        self.write("aelab25_app/second.py", "def rollback_old_second():\n    return 2\n")
        self.commit()
        self.refresh()
        before = code_index.index_status(self.index)
        old_first, old_second = self.search("rollback_old_first"), self.search("rollback_old_second")
        self.write("aelab25_app/first.py", "def rollback_new_first():\n    return 3\n")
        self.write("aelab25_app/second.py", "def rollback_new_second():\n    return 4\n")
        self.commit("Changed sources before a synthetic storage failure")
        store_file = code_index._store_file
        calls = []

        def fail_during_second_store(*args, **kwargs):
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("Injected index write failure")
            return store_file(*args, **kwargs)

        with mock.patch.object(code_index, "_store_file", side_effect=fail_during_second_store):
            with self.assertRaisesRegex((RuntimeError, ValueError), "Injected index write failure"):
                self.refresh()

        self.assertEqual(len(calls), 2)
        after = code_index.index_status(self.index)
        for key in ("files", "chunks", "source_hash", "config_hash", "revision", "commit", "dirty"):
            self.assertEqual(after[key], before[key], key)
        self.assertEqual(self.search("rollback_old_first"), old_first)
        self.assertEqual(self.search("rollback_old_second"), old_second)
        self.assertEqual(self.search("rollback_new_first"), [])
        self.assertEqual(self.search("rollback_new_second"), [])

    def test_missing_source_refresh_preserves_existing_index(self):
        self.write("aelab25_app/existing.py", "def existing_cache_marker():\n    return 1\n")
        self.commit()
        self.refresh()
        before = self.search("existing_cache_marker")

        with self.assertRaises((OSError, ValueError)):
            code_index.refresh_index(self.folder / "missing_source", self.index)
        self.assertEqual(self.search("existing_cache_marker"), before)

    def test_first_refresh_failure_does_not_prevent_a_later_successful_refresh(self):
        self.write("aelab25_app/initial.py", "def initial_recovery_marker():\n    return 1\n")
        self.commit()
        store_file = code_index._store_file

        def fail_after_real_write(*args, **kwargs):
            store_file(*args, **kwargs)
            raise RuntimeError("Injected first-refresh failure after writing source rows")

        with mock.patch.object(code_index, "_store_file", side_effect=fail_after_real_write):
            with self.assertRaisesRegex((RuntimeError, ValueError), "Injected first-refresh failure"):
                self.refresh()

        # A failed initial transaction must not strand an empty, incompatible
        # database which would require the user to find and delete the cache.
        recovered = self.refresh()
        self.assertTrue(recovered["available"])
        self.assertEqual(recovered["files"], 1)
        self.assertTrue(self.search("initial_recovery_marker"))
        self.assertEqual(code_index.index_status(self.index)["chunks"], recovered["chunks"])

    def test_unchanged_refresh_does_not_rechunk_identical_source_bytes(self):
        self.write("aelab25_app/cached.py", "def unchanged_chunk_marker():\n    return 1\n")
        self.commit()
        self.refresh()
        before = self.search("unchanged_chunk_marker")

        with mock.patch.object(code_index, "_chunks", side_effect=AssertionError("Identical source was re-chunked")) as chunker:
            report = self.refresh()

        chunker.assert_not_called()
        self.assertEqual(report["changed_count"], 0)
        self.assertEqual(report["unchanged"], ["aelab25_app/cached.py"])
        self.assertEqual(self.search("unchanged_chunk_marker"), before)

    def test_changed_policy_hash_rechunks_unchanged_source(self):
        self.write("aelab25_app/policy.py",
                   "# Synthetic multi-line source\n"
                   "def policy_refresh_marker():\n"
                   "    first = 1\n"
                   "    second = 2\n"
                   "    third = 3\n"
                   "    return first + second + third\n")
        self.commit()
        initial = self.refresh()
        new_hash = hashlib.sha256(b"synthetic revised chunk policy").hexdigest()

        with (mock.patch.object(code_index, "_CONFIG_HASH", new_hash),
              mock.patch.object(code_index, "MAX_CHUNK_LINES", 2),
              mock.patch.object(code_index, "OVERLAP_LINES", 0),
              mock.patch.object(code_index, "_chunks", wraps=code_index._chunks) as chunker):
            rebuilt = self.refresh()
            self.assertTrue(self.search("policy_refresh_marker"))

        chunker.assert_called_once()
        self.assertEqual(rebuilt["changed"], ["aelab25_app/policy.py"])
        self.assertEqual(rebuilt["unchanged_count"], 0)
        self.assertEqual(rebuilt["source_hash"], initial["source_hash"])
        self.assertEqual(rebuilt["config_hash"], new_hash)
        self.assertGreater(rebuilt["chunks"], initial["chunks"])

    def test_status_records_parseable_utc_indexing_timestamp(self):
        self.write("aelab25_app/timestamp.py", "def timestamp_marker():\n    return 1\n")
        self.commit()
        started = datetime.now(timezone.utc)
        report = self.refresh()
        finished = datetime.now(timezone.utc)

        recorded = report["indexed_at"]
        self.assertIsInstance(recorded, str)
        parsed = datetime.fromisoformat(recorded.replace("Z", "+00:00"))
        self.assertIsNotNone(parsed.tzinfo)
        self.assertEqual(parsed.utcoffset(), timedelta(0))
        # Permit serialization at whole-second precision.
        self.assertGreaterEqual(parsed, started - timedelta(seconds=1))
        self.assertLessEqual(parsed, finished + timedelta(seconds=1))
        self.assertEqual(code_index.index_status(self.index)["indexed_at"], recorded)

    def test_quoted_mapping_credentials_in_code_and_markdown_are_excluded(self):
        self.write("aelab25_app/safe.py", "def safe_mapping_index_probe():\n    return 1\n")
        fixtures = {
            "aelab25_app/configuration.py": 'settings = {"password": "syntheticTestPassword87!"}\n',
            "aelab25_app/options.py": "settings = {'access_token': 'syntheticAccessCredential87'}\n",
            "Docs/configuration.md": '```json\n{"password": "syntheticTestPassword87!"}\n```\n',
            "Docs/options.md": "```python\n{'access_token': 'syntheticAccessCredential87'}\n```\n",
        }
        for path, content in fixtures.items():
            self.write(path, content)
        self.commit()
        report = self.refresh()

        self.assertEqual(report["files"], 1)
        excluded = {item["path"] for item in report["excluded"]}
        self.assertTrue(set(fixtures).issubset(excluded))
        self.assertEqual(self.search("syntheticTestPassword87"), [])
        self.assertEqual(self.search("syntheticAccessCredential87"), [])
        self.assertTrue(self.search("safe_mapping_index_probe"))

    def test_private_key_headers_and_database_url_credentials_are_excluded(self):
        self.write("aelab25_app/safe.py", "def safe_key_index_probe():\n    return 1\n")
        fixtures = {}
        for index, kind in enumerate(("RSA", "EC", "OPENSSH")):
            fixtures[f"Docs/key_example_{index}.md"] = (
                f"-----BEGIN {kind} PRIVATE KEY-----\n"
                "syntheticnonkeyfixturecontent87\n"
                f"-----END {kind} PRIVATE KEY-----\n"
            )
        for index, scheme in enumerate(("postgresql", "mysql", "mssql+pyodbc")):
            fixtures[f"aelab25_app/database_example_{index}.py"] = (
                f'DATABASE_URL = "{scheme}://synthetic_user:syntheticTestPassword87@db.invalid/study"\n'
            )
        for path, content in fixtures.items():
            self.write(path, content)
        self.commit()
        report = self.refresh()

        self.assertEqual(report["files"], 1)
        self.assertTrue(set(fixtures).issubset({item["path"] for item in report["excluded"]}))
        self.assertEqual(self.search("syntheticTestPassword87"), [])
        self.assertEqual(self.search("syntheticnonkeyfixturecontent87"), [])
        self.assertTrue(self.search("safe_key_index_probe"))

    def test_environment_lookups_and_explicit_dummy_credentials_remain_allowed(self):
        fixtures = {
            "aelab25_app/environment.py": (
                "import os\n"
                'password = os.environ["PASSWORD"]\n'
                'access_token = os.environ.get("ACCESS_TOKEN")\n'
                'database_url = os.getenv("DATABASE_URL")\n'
                "def environment_lookup_probe():\n    return None\n"
            ),
            "aelab25_app/dummy_options.py": (
                'settings = {"password": "dummy", "access_token": "example"}\n'
                "def explicit_dummy_probe():\n    return settings\n"
            ),
            "Docs/dummy_guide.md": 'Example configuration:\n{"password": "dummy", "access_token": "placeholder"}\nexplicit_dummy_doc_probe\n',
        }
        for path, content in fixtures.items():
            self.write(path, content)
        self.commit()
        report = self.refresh()

        self.assertEqual(report["files"], len(fixtures))
        for marker in ("environment_lookup_probe", "explicit_dummy_probe", "explicit_dummy_doc_probe"):
            self.assertTrue(self.search(marker))

    def test_scanner_policy_upgrade_prunes_previously_indexed_credentials(self):
        unsafe_path = "aelab25_app/old_configuration.py"
        self.write(unsafe_path,
                   'settings = {"access_token": "syntheticUpgradeCredential87"}\n'
                   "def old_scanner_exposure_probe():\n    return settings\n")
        self.write("aelab25_app/retained.py", "def retained_upgrade_probe():\n    return 1\n")
        self.commit()
        old_policy = hashlib.sha256(b"synthetic old policy without mapping detection").hexdigest()
        with (mock.patch.object(code_index, "_CONFIG_HASH", old_policy),
              mock.patch.object(code_index, "_has_secret", return_value=False)):
            original = self.refresh()
            self.assertEqual(original["files"], 2)
            self.assertTrue(self.search("old_scanner_exposure_probe"))

        refreshed = self.refresh()

        self.assertEqual(refreshed["files"], 1)
        self.assertIn(unsafe_path, refreshed["deleted"])
        self.assertIn(unsafe_path, {item["path"] for item in refreshed["excluded"]})
        self.assertEqual(self.search("old_scanner_exposure_probe"), [])
        self.assertEqual(self.search("syntheticUpgradeCredential87"), [])
        self.assertTrue(self.search("retained_upgrade_probe"))
        # Verify no active copied text or search row remains, independent of
        # ranking/query behavior. This is logical pruning, not forensic erasure.
        with closing(sqlite3.connect(self.index)) as connection:
            for table, key in (("files", "path"), ("chunks", "source_path"), ("chunks_fts", "source_path")):
                self.assertEqual(connection.execute(f"SELECT count(*) FROM {table} WHERE {key} = ?", (unsafe_path,)).fetchone()[0], 0)

    def test_old_policy_is_rejected_until_refresh_migrates_the_index(self):
        self.write("aelab25_app/policy.py", "def migration_guard_probe():\n    return 1\n")
        self.commit()
        self.refresh()
        new_policy = hashlib.sha256(b"synthetic replacement security policy").hexdigest()

        with mock.patch.object(code_index, "_CONFIG_HASH", new_policy):
            with self.assertRaisesRegex(ValueError, "policy|refresh|rebuild"):
                code_index.index_status(self.index)
            with self.assertRaisesRegex(ValueError, "policy|refresh|rebuild"):
                self.search("migration_guard_probe")
            migrated = self.refresh()
            self.assertEqual(migrated["config_hash"], new_policy)
            self.assertTrue(code_index.index_status(self.index)["available"])
            self.assertTrue(self.search("migration_guard_probe"))

    def test_source_edit_during_refresh_aborts_and_preserves_previous_snapshot(self):
        for initially_dirty in (False, True):
            with self.subTest(initially_dirty=initially_dirty):
                path = self.write("aelab25_app/concurrent.py", "def concurrent_original_probe():\n    return 1\n")
                self.commit("Snapshot before concurrent-edit test")
                self.refresh()
                before = code_index.index_status(self.index)
                prior_hits = self.search("concurrent_original_probe")
                self.write("aelab25_app/concurrent.py", "def intended_refresh_probe():\n    return 2\n")
                if not initially_dirty:
                    self.commit("Clean source snapshot to index")
                store_file = code_index._store_file

                def edit_after_store(*args, **kwargs):
                    store_file(*args, **kwargs)
                    path.write_text("def concurrent_unseen_probe():\n    return 3\n", encoding="utf-8")

                with mock.patch.object(code_index, "_store_file", side_effect=edit_after_store):
                    with self.assertRaisesRegex(ValueError, "changed|rerun|retry"):
                        self.refresh()

                after = code_index.index_status(self.index)
                for key in ("source_hash", "revision", "files", "chunks", "indexed_at"):
                    self.assertEqual(after[key], before[key], key)
                self.assertEqual(self.search("concurrent_original_probe"), prior_hits)
                self.assertEqual(self.search("intended_refresh_probe"), [])
                self.assertEqual(self.search("concurrent_unseen_probe"), [])

    def test_configured_study_channel_files_and_flatstart_tree_are_excluded(self):
        excluded = (
            "Flatstart/scenario_one/prepare.py",
            "Flatstart/scenario_one/notes.md",
            "PEC_Add_Channels.py",
            "ERCOT_Add_Channels.py",
            "TRABAJOv6a_Install/TRABAJO/ADD_Channels.py",
            "TRABAJOv6a_Install/TRABAJO/ADD_Channels_PEC.py",
            "TRABAJOv6a_Install/TRABAJO/ADD_Channels_PEC - Copy.py",
            "TRABAJOv6a_Install/TRABAJO/ADD_Channels_TEMPLATE.py",
            "TRABAJOv6a_Install/TRABAJO/ADD_Channels_CustomStudy.py",
        )
        allowed = {
            "aelab25_app/dynamic_channels.py": "def generic_channel_capture_probe():\n    return 1\n",
            "MakeCnvAndSnap35.py": "def generic_case_preparation_probe():\n    return 2\n",
            "TRABAJO/Python/MakeCnvAndSnap35.py": "def generic_trabajo_preparation_probe():\n    return 3\n",
        }
        for path in excluded:
            # Synthetic marker only: no real bus numbers or study data copied.
            self.write(path, "syntheticStudyOnlyMarker987\n")
        for path, content in allowed.items():
            self.write(path, content)
        self.commit()
        report = self.refresh()

        self.assertEqual(report["files"], len(allowed))
        self.assertTrue(set(excluded).issubset({item["path"] for item in report["excluded"]}))
        self.assertEqual(self.search("syntheticStudyOnlyMarker987"), [])
        for marker in ("generic_channel_capture_probe", "generic_case_preparation_probe", "generic_trabajo_preparation_probe"):
            self.assertTrue(self.search(marker))
        self.assertNotIn("syntheticStudyOnlyMarker987", str(report))


if __name__ == "__main__":
    unittest.main()
