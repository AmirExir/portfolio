import json
import numpy as np
from sentence_transformers import SentenceTransformer

# === Step 1: Load from chunks.json and write to ercot_chunks_cached.json ===
with open("chunks.json", "r") as f:
    chunks = json.load(f)

with open("ercot_chunks_cached.json", "w") as f:
    json.dump(chunks, f)

print("✅ Step 1: ercot_chunks_cached.json created.")

# === Step 2: Generate Embeddings ===
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [chunk["content"] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# === Step 3: Save embeddings to .npy file ===
np.save("chatbot_ercot_all_in_one/ercot_embeddings.npy", embeddings)
print(f"✅ Step 2: Total embeddings created: {len(embeddings)}")
print("✅ Step 3: Embeddings saved to ercot_embeddings.npy")