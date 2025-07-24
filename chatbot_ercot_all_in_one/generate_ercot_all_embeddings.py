import os
import json
import numpy as np
from openai import OpenAI
from typing import List
import time

# Load your ERCOT document chunks (adjust the file if needed)
with open("ercot_combined_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Retry-safe OpenAI embedding call
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

# Compute embeddings
embeddings = []
embedding_model = "text-embedding-3-large"  # or "text-embedding-3-small"

for chunk in chunks:
    text = chunk["text"][:8192]  # token limit
    response = safe_openai_call(
        client.embeddings.create,
        model=embedding_model,
        input=text
    )
    if response and response.data:
        embeddings.append(response.data[0].embedding)
    else:
        print(f"❌ Skipped chunk {chunk.get('id')} due to error")
        embeddings.append(None)

# Filter out any failed embeddings
valid_data = [(c, e) for c, e in zip(chunks, embeddings) if e is not None]
if not valid_data:
    raise RuntimeError("No valid embeddings generated.")

final_chunks, final_embeddings = zip(*valid_data)

# Save to disk
np.save("ercot_embeddings.npy", np.array(final_embeddings))
with open("ercot_chunks_cached.json", "w", encoding="utf-8") as f:
    json.dump(final_chunks, f, indent=2)

print("✅ Embeddings and chunks saved!")