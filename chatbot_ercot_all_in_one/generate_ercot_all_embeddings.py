import argparse
import json
import numpy as np
import os
import time
import re
from pathlib import Path
from openai import OpenAI

# === Setup ===
base_dir = Path(__file__).resolve().parent
source_dir = base_dir / "ercot_sources"
chunk_output_file = base_dir / "ercot_chunks_cached.json"
embedding_output_file = base_dir / "ercot_embeddings.npy"
chunk_size = 7600  # character-based, kept below the per-call truncation guard
chunk_overlap = 400
embedding_model = "text-embedding-3-large"

parser = argparse.ArgumentParser(description="Regenerate ERCOT RAG chunks and OpenAI embeddings.")
parser.add_argument("--dry-run", action="store_true", help="Validate chunking without calling OpenAI or writing files.")
args = parser.parse_args()

# === Step 1: Load and paragraph-aware chunk all ERCOT .txt files ===
chunks = []

def split_long_paragraph(text, max_size):
    if len(text) <= max_size:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_size)
        if end < len(text):
            boundary = max(
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("; ", start, end),
            )
            if boundary > start + (max_size // 2):
                end = boundary + 1
        part = text[start:end].strip()
        if part:
            parts.append(part)
        start = end
    return parts

def chunk_paragraphs(text, chunk_size, overlap):
    paragraphs = re.split(r"\n\s*\n", text)  # split on empty lines
    current_chunk = ""
    result_chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        for piece in split_long_paragraph(para, chunk_size):
            if len(current_chunk) + len(piece) + 2 <= chunk_size:
                current_chunk += piece + "\n\n"
            else:
                if current_chunk.strip():
                    result_chunks.append(current_chunk.strip())
                current_chunk = piece + "\n\n"
    if current_chunk:
        result_chunks.append(current_chunk.strip())
    
    # Add overlap
    final_chunks = []
    for i, chunk in enumerate(result_chunks):
        overlap_text = result_chunks[i - 1][-overlap:] if i > 0 else ""
        combined = (overlap_text + "\n" + chunk).strip()
        final_chunks.append(combined)
    return final_chunks

source_files = sorted(source_dir.glob("*.txt"))
for filepath in source_files:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
        chunk_texts = chunk_paragraphs(text, chunk_size, chunk_overlap)
        for idx, chunk_text in enumerate(chunk_texts):
            chunks.append({
                "text": chunk_text,
                "source": filepath.name,
                "chunk_index": idx
            })

print(f" Loaded and chunked {len(chunks)} chunks from {len(source_files)} files.")
max_chunk_len = max((len(chunk["text"]) for chunk in chunks), default=0)
oversized_chunks = [
    (idx, chunk["source"], chunk["chunk_index"], len(chunk["text"]))
    for idx, chunk in enumerate(chunks)
    if len(chunk["text"]) > 8192
]
section_9_chunks = [
    idx
    for idx, chunk in enumerate(chunks)
    if chunk["source"] == "ercotaiassistant.txt"
    and "LARGE LOAD ADDITIONS AT NEW OR MODIFICATION" in chunk["text"]
]

print(f" Max chunk length: {max_chunk_len} characters")
print(f" ERCOT Planning Guide Section 9 chunks: {section_9_chunks}")

if oversized_chunks:
    preview = ", ".join(str(item) for item in oversized_chunks[:5])
    raise RuntimeError(f"Found chunks over 8192 characters after splitting: {preview}")

if args.dry_run:
    print(" Dry run complete. No files were written.")
    raise SystemExit(0)

# === Step 2: Initialize OpenAI ===
api_key = os.getenv("OPENAI_API_KEY", "").strip()
if not api_key or api_key in {"your-key", "your-key-here"}:
    raise SystemExit(
        "OPENAI_API_KEY is missing or still set to a placeholder. "
        "Export your real key before running this script."
    )

client = OpenAI(api_key=api_key)

def safe_openai_call(api_function, max_retries=5, backoff_factor=2, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return api_function(**kwargs)
        except Exception as e:
            if getattr(e, "status_code", None) in {401, 403}:
                raise RuntimeError(
                    "OpenAI authentication failed. Check OPENAI_API_KEY; not retrying."
                ) from None
            wait_time = backoff_factor ** retries
            print(f" Error: {e} — Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    return None

# === Step 3: Compute embeddings ===
embeddings = []
for i, chunk in enumerate(chunks):
    text = chunk["text"]
    print(f" Processing chunk {i+1}/{len(chunks)}")
    response = safe_openai_call(
        client.embeddings.create,
        model=embedding_model,
        input=text
    )
    if response and response.data:
        embeddings.append(response.data[0].embedding)
        print(f" Chunk {i+1} embedded")
    else:
        print(f" Skipped chunk {i+1} due to error")
        embeddings.append(None)

# === Step 4: Filter out failed ===
valid_data = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
if not valid_data:
    raise RuntimeError("No valid embeddings generated.")
final_chunks, final_embeddings = zip(*valid_data)

# === Step 5: Save ===
tmp_chunk_output_file = chunk_output_file.with_suffix(".json.tmp")
tmp_embedding_output_file = embedding_output_file.with_suffix(".npy.tmp")

with open(tmp_chunk_output_file, "w", encoding="utf-8") as f:
    json.dump(final_chunks, f, indent=2)
with open(tmp_embedding_output_file, "wb") as f:
    np.save(f, np.array(final_embeddings))

tmp_chunk_output_file.replace(chunk_output_file)
tmp_embedding_output_file.replace(embedding_output_file)

print(f"\n Saved {len(final_chunks)} valid chunks to:")
print(f"   → {chunk_output_file}")
print(f"   → {embedding_output_file}")
