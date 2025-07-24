# make_ercot_embeddings.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Load cached chunks (make sure this file exists)
with open("chatbot_ercot_all_in_one/ercot_chunks_cached.json", "r") as f:
    chunks = json.load(f)

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
texts = [chunk["content"] for chunk in chunks]
embeddings = model.encode(texts, show_progress_bar=True)

# Save to file
np.save("chatbot_ercot_all_in_one/ercot_embeddings.npy", embeddings)

print("✅ Done: Embeddings saved to ercot_embeddings.npy")