import os
import json
import numpy as np
import time
from openai import OpenAI

# === Setup ===
source_dir = "ercot_sources"
chunk_output_file = "ercot_chunks_cached.json"
embedding_output_file = "ercot_embeddings.npy"
chunk_size = 800
embedding_model = "text-embedding-3-large"

# === Step 1: Load and chunk all ERCOT .txt files ===
chunks = []
for filename in os.listdir(source_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(source_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i+chunk_size].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "source": filename,
                        "chunk_index": i // chunk_size
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
    text = chunk["text"][:8192]
    response = safe_openai_call(
        client.embeddings.create,
        model=embedding_model,
        input=text
    )
    if response and response.data:
        embeddings.append(response.data[0].embedding)
    else:
        print(f"❌ Skipped chunk {i} due to error")
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