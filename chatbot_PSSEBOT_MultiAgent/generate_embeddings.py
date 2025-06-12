from retriever import load_chunks_and_embeddings
import numpy as np
import json

chunks, embeddings = load_chunks_and_embeddings()

# Save to disk
np.save("psse_embeddings.npy", embeddings)
with open("psse_chunks_cached.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print("✅ Embeddings saved to psse_embeddings.npy and psse_chunks_cached.json")