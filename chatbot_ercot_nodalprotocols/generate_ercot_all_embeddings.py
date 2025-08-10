import os
import json
import numpy as np
import time
import re
from openai import OpenAI

# === Setup ===
source_dir = "ercot_sources"
chunk_output_file = "ercot_chunks_cached.json"
embedding_output_file = "ercot_embeddings.npy"
chunk_size = 8000  # character-based for now (not token)
chunk_overlap = 500
embedding_model = "text-embedding-3-large"

# === Step 1: Load and paragraph-aware chunk all ERCOT .txt files ===
chunks = []

def chunk_paragraphs(text, chunk_size, overlap):
    paragraphs = re.split(r"\n\s*\n", text)  # split on empty lines
    current_chunk = ""
    result_chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += para + "\n\n"
        else:
            result_chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk:
        result_chunks.append(current_chunk.strip())
    
    # Add overlap
    final_chunks = []
    for i, chunk in enumerate(result_chunks):
        overlap_text = result_chunks[i - 1][-overlap:] if i > 0 else ""
        combined = (overlap_text + "\n" + chunk).strip()
        final_chunks.append(combined)
    return final_chunks

for filename in os.listdir(source_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(source_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            chunk_texts = chunk_paragraphs(text, chunk_size, chunk_overlap)
            for idx, chunk_text in enumerate(chunk_texts):
                chunks.append({
                    "text": chunk_text,
                    "source": filename,
                    "chunk_index": idx
                })

print(f"✅ Loaded and chunked {len(chunks)} chunks from {len(os.listdir(source_dir))} files.")

# === Step 2: Initialize OpenAI ===
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def safe_openai_call(api_function, max_retries=5, backoff_factor=2, **kwargs):
    retries = 0
    while retries < max_retries:
        try:
            return api_function(**kwargs)
        except Exception as e:
            wait_time = backoff_factor ** retries
            print(f"⚠️ Error: {e} — Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    return None

# === Step 3: Compute embeddings ===
embeddings = []
for i, chunk in enumerate(chunks):
    text = chunk["text"][:8192]  # API limit
    print(f"🔄 Processing chunk {i+1}/{len(chunks)}")
    response = safe_openai_call(
        client.embeddings.create,
        model=embedding_model,
        input=text
    )
    if response and response.data:
        embeddings.append(response.data[0].embedding)
        print(f"✅ Chunk {i+1} embedded")
    else:
        print(f"❌ Skipped chunk {i+1} due to error")
        embeddings.append(None)

# === Step 4: Filter out failed ===
valid_data = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
if not valid_data:
    raise RuntimeError("No valid embeddings generated.")
final_chunks, final_embeddings = zip(*valid_data)

# === Step 5: Save ===
with open(chunk_output_file, "w", encoding="utf-8") as f:
    json.dump(final_chunks, f, indent=2)
np.save(embedding_output_file, np.array(final_embeddings))

print(f"\n✅ Saved {len(final_chunks)} valid chunks to:")
print(f"   → {chunk_output_file}")
print(f"   → {embedding_output_file}")