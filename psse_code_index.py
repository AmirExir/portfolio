"""Incremental local repository-code retrieval; never executes or uploads code.

The SQLite FTS5 index is separate from both assistants' existing document
embeddings. Refresh reads only tracked, eligible source files. A transaction
publishes the complete new code snapshot or leaves the previous index intact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import tokenize
from typing import Any


SCHEMA_VERSION = 1
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_CHUNK_LINES = 80
MAX_CHUNK_CHARS = 6000
OVERLAP_LINES = 8
_EXCLUDED_PARTS = frozenset({
    "tests", "test", "vendor", "vendors", "third_party", "third-party",
    "node_modules", "site-packages", ".venv", "venv", "env", ".git",
    ".rag", ".streamlit", "__pycache__", "codex-skills", "xlrd", "xlwt",
    "curses", "numpy", "pandas", "matplotlib", "scipy", "openpyxl",
    "events", "event", "eventss", "cases", "case", "dyrs", "chngs",
    "results", "result", "outputs", "output", "outs", "out", "logs", "log",
    "crash_logs", "libs", "lib", "dlls", "generated", "archives", "archive",
    "_con", "source_cons", "flatstart",
})
_EXCLUDED_SOURCE_FILES = frozenset({
    "pec_add_channels.py", "ercot_add_channels.py",
})
_POLICY = {
    "version": 2, "extensions": [".py", ".md"],
    "excluded_parts": sorted(_EXCLUDED_PARTS),
    "excluded_source_files": sorted(_EXCLUDED_SOURCE_FILES),
    "excluded_source_patterns": ["TRABAJOv6a_Install/TRABAJO/ADD_Channels*.py"],
    "excluded_roots": ["dmviwe", "pypowergem*"],
    "chunk_lines": MAX_CHUNK_LINES, "chunk_chars": MAX_CHUNK_CHARS,
    "overlap_lines": OVERLAP_LINES, "max_file_bytes": MAX_FILE_BYTES,
    "secret_scan": "literal-credentials-v2",
}
_CONFIG_HASH = hashlib.sha256(json.dumps(_POLICY, sort_keys=True).encode()).hexdigest()
_SECRET_ASSIGNMENT = re.compile(
    r"(?im)\b[\w]*(?:api[_-]?key|password|passwd|secret|access[_-]?token|auth[_-]?token)\b"
    r"['\"]?\s*[:=]\s*(['\"])([^'\"\r\n]+)\1"
)
_SECRET_TOKEN = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b")
_STOPWORDS = frozenset("a an and are as at be by can code do does for from how i in is it of on or please python the this to use using what where which with".split())


def default_index_path() -> Path:
    """Return the configured local index, or the adjacent ignored .rag folder."""
    configured = os.environ.get("PSSE_CODE_INDEX", "").strip()
    return Path(configured).expanduser() if configured else Path(__file__).resolve().parent / ".rag" / "psse_code.sqlite3"


def _git(root: Path, *args: str, required: bool = True) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False)
    if result.returncode and required:
        raise ValueError("Cannot read the source Git repository; select its repository root.")
    return result.stdout if result.returncode == 0 else b""


def _exclusion(path: str) -> str | None:
    parts = Path(path).parts
    lowered = [part.casefold() for part in parts]
    if not parts or Path(path).is_absolute() or ".." in parts:
        return "invalid repository-relative path"
    if lowered[0] == "dmviwe" or lowered[0].startswith("pypowergem"):
        return "bundled non-PSS/E vendor source"
    if ("/".join(lowered) in _EXCLUDED_SOURCE_FILES
            or (lowered[:-1] == ["trabajov6a_install", "trabajo"]
                and lowered[-1].startswith("add_channels") and lowered[-1].endswith(".py"))):
        return "configured study channel source"
    if any(part in _EXCLUDED_PARTS or part.startswith("psutil") for part in lowered[:-1]):
        return "dependency, test, study-data, or generated directory"
    name = lowered[-1]
    if name == "agents.md" or name.startswith("test_") or name.endswith("_test.py"):
        return "agent instructions or test source"
    if name.startswith(".env") or any(word in name for word in ("credential", "secrets")):
        return "credential/configuration file"
    if Path(name).suffix not in {".py", ".md"}:
        return "not Python or engineering Markdown"
    return None


def _decode(raw: bytes, suffix: str) -> tuple[str, str]:
    if suffix == ".py":
        try:
            encoding, _ = tokenize.detect_encoding(io.BytesIO(raw).readline)
        except SyntaxError:
            # Legacy Python 2 source without a cookie can contain cp1252 text.
            # An explicit, invalid declaration must never be silently ignored.
            if re.search(br"coding\s*[:=]", b"\n".join(raw.splitlines()[:2])):
                raise ValueError("invalid Python source encoding declaration")
            encoding = "cp1252"
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            if re.search(br"coding\s*[:=]", b"\n".join(raw.splitlines()[:2])):
                raise ValueError("source bytes disagree with declared encoding")
            return raw.decode("cp1252"), "cp1252"
    try:
        return raw.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        return raw.decode("cp1252"), "cp1252"


def _has_secret(text: str) -> bool:
    if (_SECRET_TOKEN.search(text)
            or re.search(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----", text)
            or re.search(r"(?i)\b[a-z][a-z0-9+.-]*://[^/\s:@]+:[^@\s/]+@", text)):
        return True
    for match in _SECRET_ASSIGNMENT.finditer(text):
        value = match.group(2).strip()
        placeholder = (not value or value.startswith(("${", "<", "YOUR_", "your_"))
                       or set(value) <= {"*", "x", "X"}
                       or value.casefold() in {"placeholder", "changeme", "example", "dummy", "redacted"})
        if not placeholder:
            return True
    # Connection strings sometimes contain credentials inside one string.
    return bool(re.search(r"(?i)[;'\"]\s*(?:PWD|PASSWORD)\s*=\s*[^;'\"\s{}][^;'\"\r\n]{3,}", text))


def _chunks(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines(keepends=True)
    result: list[dict[str, Any]] = []
    start = 0
    while start < len(lines):
        if len(lines[start]) > MAX_CHUNK_CHARS:
            line = lines[start].rstrip("\r\n")
            for offset in range(0, len(line), MAX_CHUNK_CHARS):
                result.append(dict(start_line=start + 1, end_line=start + 1,
                                   start_column=offset + 1, end_column=min(offset + MAX_CHUNK_CHARS, len(line)),
                                   source_text=line[offset:offset + MAX_CHUNK_CHARS]))
            start += 1
            continue
        end, characters = start, 0
        while end < len(lines) and end - start < MAX_CHUNK_LINES:
            if characters + len(lines[end]) > MAX_CHUNK_CHARS:
                break
            characters += len(lines[end])
            end += 1
        excerpt = "".join(lines[start:end]).rstrip("\r\n")
        if excerpt.strip():
            result.append(dict(start_line=start + 1, end_line=end, start_column=None,
                               end_column=None, source_text=excerpt))
        if end == len(lines):
            break
        start = max(start + 1, end - OVERLAP_LINES)
    return result


def _connect_read(index_path: Path) -> sqlite3.Connection:
    try:
        return sqlite3.connect(index_path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    except sqlite3.Error as exc:
        raise ValueError("Cannot open the local code index; choose a new --index path or move the damaged cache before refreshing.") from exc


def _metadata(connection: sqlite3.Connection, *, require_current_policy: bool = True) -> dict:
    try:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if int(metadata.get("schema_version", "0")) != SCHEMA_VERSION:
            raise ValueError("unsupported schema")
        result = json.loads(metadata["status"])
        files = connection.execute("SELECT count(*) FROM files").fetchone()[0]
        chunks = connection.execute("SELECT count(*) FROM chunks").fetchone()[0]
        if result["files"] != files or result["chunks"] != chunks:
            raise ValueError("record counts disagree")
        if connection.execute("SELECT count(*) FROM chunks_fts").fetchone()[0] != chunks:
            raise ValueError("search rows disagree")
    except (sqlite3.Error, ValueError, TypeError, KeyError) as exc:
        raise ValueError("The local code index is corrupt or incompatible; choose a new --index path or move the damaged cache before refreshing.") from exc
    if require_current_policy and result.get("config_hash") != _CONFIG_HASH:
        raise ValueError("The local code index uses an older indexing policy; run the offline refresh before using it.")
    return result


def index_status(index_path: Path) -> dict:
    """Read provenance/counts; missing is explicit and corrupt is an error."""
    path = Path(index_path)
    if not path.exists():
        return {"available": False, "files": 0, "chunks": 0, "revision": "", "index_path": str(path)}
    connection = _connect_read(path)
    try:
        connection.execute("BEGIN")  # Keep validation reads in one snapshot.
        return dict(_metadata(connection), available=True, index_path=str(path))
    finally:
        connection.close()


def _store_file(connection: sqlite3.Connection, path: str, file_sha256: str,
                encoding: str, chunks: list[dict]) -> None:
    """Replace one file inside the caller's all-or-nothing transaction."""
    connection.execute("DELETE FROM chunks_fts WHERE source_path = ?", (path,))
    connection.execute("DELETE FROM chunks WHERE source_path = ?", (path,))
    connection.execute("INSERT OR REPLACE INTO files VALUES (?, ?, ?)", (path, file_sha256, encoding))
    for chunk in chunks:
        identifier = f"psse_codes:{path}:L{chunk['start_line']}-L{chunk['end_line']}"
        if chunk["start_column"] is not None:
            identifier += f":C{chunk['start_column']}-{chunk['end_column']}"
        connection.execute("INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (identifier, path, chunk["start_line"], chunk["end_line"],
                            chunk["start_column"], chunk["end_column"], chunk["source_text"]))
        connection.execute("INSERT INTO chunks_fts VALUES (?, ?, ?)", (identifier, path, chunk["source_text"]))


def refresh_index(source_root: Path, index_path: Path) -> dict:
    """Index eligible tracked sources locally; unchanged source rows are reused.

    Reports changed/unchanged/deleted relative paths and excluded paths/reasons.
    Source bytes, not just Git HEAD, determine the snapshot and file hashes.
    No source file is imported, executed, modified, or sent over the network.
    """
    root = Path(source_root).expanduser().resolve()
    git_root = _git(root, "rev-parse", "--show-toplevel").decode().strip()
    if Path(git_root).resolve() != root:
        raise ValueError("Select the source Git repository root, not a subdirectory.")
    tracked = sorted(os.fsdecode(value) for value in _git(root, "ls-files", "-z").split(b"\0") if value)
    commit = _git(root, "rev-parse", "--verify", "HEAD", required=False).decode().strip() or "uncommitted"
    git_status = _git(root, "status", "--porcelain", "--untracked-files=no")
    dirty = bool(git_status)
    revision = commit + ("-dirty" if dirty else "")
    sources, excluded, exclusion_counts = {}, [], Counter()
    for path in tracked:
        reason = _exclusion(path)
        if reason is None:
            try:
                original = root / path
                resolved = original.resolve(strict=True)
                if original.is_symlink() or not resolved.is_relative_to(root):
                    raise ValueError("symlink or path escapes are not indexed")
                with resolved.open("rb") as stream:
                    raw = stream.read(MAX_FILE_BYTES + 1)
                if len(raw) > MAX_FILE_BYTES:
                    raise ValueError("source file exceeds the size limit")
                text, encoding = _decode(raw, original.suffix.casefold())
                if "\x00" in text:
                    raise ValueError("binary content in a source file")
                if _has_secret(text):
                    raise ValueError("possible hardcoded credential; file skipped")
                if not text.strip():
                    raise ValueError("empty source file")
                sources[path] = (hashlib.sha256(raw).hexdigest(), encoding, text)
            except (OSError, UnicodeError, ValueError) as exc:
                # Never copy file text or decoding exception snippets to reports.
                reason = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, UnicodeError) else "source unavailable or undecodable"
        if reason is not None:
            exclusion_counts[reason] += 1
            # Study filenames and other non-code inventory do not belong in the
            # code cache. Their exclusions are recorded only as aggregate counts.
            if Path(path).suffix.casefold() in {".py", ".md"}:
                excluded.append({"path": path, "reason": reason})
    if not sources:
        raise ValueError("No eligible tracked source files were found; the previous code index was preserved.")
    path = Path(index_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    connection = sqlite3.connect(path, timeout=10)
    committed = False
    try:
        connection.execute("BEGIN IMMEDIATE")
        if existed:
            previous = _metadata(connection, require_current_policy=False)
            if previous.get("source_root") != str(root):
                raise ValueError("This code index belongs to a different source repository; select a separate index path.")
        else:
            connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE files(path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, encoding TEXT NOT NULL)")
            connection.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY, source_path TEXT NOT NULL, start_line INTEGER, end_line INTEGER, start_column INTEGER, end_column INTEGER, text TEXT NOT NULL)")
            connection.execute("CREATE INDEX chunks_source ON chunks(source_path)")
            connection.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(id UNINDEXED, source_path, text)")
            previous = {}
        old = dict(connection.execute("SELECT path, sha256 FROM files"))
        changed, unchanged = [], []
        for source, (digest, encoding, text) in sources.items():
            if old.get(source) == digest and previous.get("config_hash") == _CONFIG_HASH:
                unchanged.append(source)
            else:
                _store_file(connection, source, digest, encoding, _chunks(text))
                changed.append(source)
        deleted = sorted(set(old) - set(sources))
        for source in deleted:
            connection.execute("DELETE FROM chunks_fts WHERE source_path = ?", (source,))
            connection.execute("DELETE FROM chunks WHERE source_path = ?", (source,))
            connection.execute("DELETE FROM files WHERE path = ?", (source,))
        source_hash = hashlib.sha256(json.dumps([(name, item[0]) for name, item in sources.items()], ensure_ascii=True).encode()).hexdigest()
        status = {"available": True, "source_root": str(root), "commit": commit, "dirty": dirty,
                  "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "revision": revision, "source_hash": source_hash, "config_hash": _CONFIG_HASH,
                  "files": len(sources), "chunks": connection.execute("SELECT count(*) FROM chunks").fetchone()[0],
                  "changed": changed, "unchanged": unchanged, "deleted": deleted, "excluded": excluded,
                  "changed_count": len(changed), "unchanged_count": len(unchanged), "deleted_count": len(deleted),
                  "excluded_count": sum(exclusion_counts.values()), "exclusion_counts": dict(exclusion_counts)}
        # A worktree already marked dirty can change again without changing
        # porcelain status. Recheck accepted bytes as well as Git provenance.
        changed_during_scan = False
        for source, (digest, _encoding, _text) in sources.items():
            try:
                original = root / source
                resolved = original.resolve(strict=True)
                if original.is_symlink() or not resolved.is_relative_to(root):
                    changed_during_scan = True
                    break
                with resolved.open("rb") as stream:
                    current = stream.read(MAX_FILE_BYTES + 1)
                if hashlib.sha256(current).hexdigest() != digest:
                    changed_during_scan = True
                    break
            except OSError:
                changed_during_scan = True
                break
        final_commit = _git(root, "rev-parse", "--verify", "HEAD", required=False).decode().strip() or "uncommitted"
        final_status = _git(root, "status", "--porcelain", "--untracked-files=no")
        if changed_during_scan or final_commit != commit or final_status != git_status:
            raise ValueError("The source Git revision or worktree changed during indexing; the previous index was preserved. Rerun the refresh after edits finish.")
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        connection.execute("INSERT OR REPLACE INTO metadata VALUES ('status', ?)", (json.dumps(status, sort_keys=True),))
        connection.commit()
        committed = True
        return dict(status, index_path=str(path))
    except sqlite3.Error as exc:
        connection.rollback()
        raise ValueError("Cannot refresh the local code index; ensure SQLite supports FTS5 and the index is writable.") from exc
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        if not existed and not committed:
            # Rollback preserves an existing index. A failed first creation has
            # no valid previous artifact and must not block the next refresh.
            path.unlink(missing_ok=True)


def search_code(query: str, index_path: Path, k: int = 4) -> list[dict]:
    """Search bounded literal terms, with source provenance and no API calls."""
    if type(k) is not int or not 1 <= k <= 20:
        raise ValueError("Code retrieval count must be an integer from 1 to 20.")
    if not isinstance(query, str):
        raise ValueError("Code retrieval query must be text.")
    # Keep identifiers and dotted API names as quoted phrases. Splitting a
    # deleted_symbol into loose OR terms would return unrelated *_symbol code.
    terms = list(dict.fromkeys(term.casefold() for term in re.findall(r"\w+(?:\.\w+)*", query[:4000])
                               if len(term) <= 64 and term.casefold() not in _STOPWORDS))[:16]
    path = Path(index_path)
    if not terms or not path.exists():
        return []
    connection = _connect_read(path)
    try:
        connection.execute("BEGIN")  # Metadata and search rows share one snapshot.
        status = _metadata(connection)
        expression = " OR ".join('"' + term + '"' for term in terms)
        rows = connection.execute(
            "SELECT c.id,c.source_path,c.start_line,c.end_line,c.start_column,c.end_column,c.text,f.sha256,"
            "bm25(chunks_fts,0,5,1) AS score FROM chunks_fts JOIN chunks c ON c.id=chunks_fts.id "
            "JOIN files f ON f.path=c.source_path WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?",
            (expression, max(40, k * 12)),
        ).fetchall()
        rows.sort(key=lambda item: (item[8] * (1.15 if item[1].startswith("aelab25_app/") else 1), item[1], item[2]))
        results, seen = [], set()
        for identifier, source, start, end, start_column, end_column, text, digest, _score in rows:
            # Adjacent overlapping excerpts from one file should not exhaust all
            # reference slots; identical snippets are also returned only once.
            identity = (source, text)
            if identity in seen or any(item["source_path"] == source and not (end < item["start_line"] or start > item["end_line"]) for item in results):
                continue
            seen.add(identity)
            location = f"psse_codes/{source}:{start}-{end}"
            if start_column is not None:
                location += f" (columns {start_column}-{end_column})"
            reference = ("Repository code example; not authoritative PSS/E API documentation.\n"
                         f"Source: {location}\nRevision: {status['revision']}\nFile SHA256: {digest}\n\n{text}")
            results.append(dict(id=identifier, text=reference, source_text=text, source_path=source,
                                start_line=start, end_line=end, start_column=start_column, end_column=end_column,
                                file_sha256=digest, revision=status["revision"], source_kind="repository_code"))
            if len(results) == k:
                break
        return results
    except sqlite3.Error as exc:
        raise ValueError("Cannot search the local code index; choose a new --index path or move the damaged cache before refreshing.") from exc
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Source Git repository root; required for refresh.")
    parser.add_argument("--index", type=Path, default=default_index_path())
    parser.add_argument("--check", action="store_true", help="Read index status without modifying it.")
    args = parser.parse_args()
    if not args.check and args.source is None:
        parser.error("--source is required unless --check is used")
    try:
        report = index_status(args.index) if args.check else refresh_index(args.source, args.index)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Code index: {exc}\n")
    # CLI summary deliberately avoids dumping source text or excluded filenames.
    print(json.dumps({key: value for key, value in report.items() if key not in {"changed", "unchanged", "deleted", "excluded"}}, indent=2))


if __name__ == "__main__":
    main()
