import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load cached chunks
with open("ercot_chunks_cached.json", "r") as f:
    chunks = json.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
texts = [chunk["content"] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# Save to file
np.save("ercot_embeddings.npy", embeddings)

print(f"🔢 Total embeddings created: {len(embeddings)}")
print("✅ Done: Embeddings saved to ercot_embeddings.npy")