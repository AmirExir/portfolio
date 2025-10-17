from retriever import load_chunks_and_embeddings
import numpy as np

# Load and embed
chunks, embeddings = load_chunks_and_embeddings("input_chunks.json")

# Save embeddings only
np.save("psse_embeddings.npy", embeddings)

print(" Embeddings saved to psse_embeddings.npy (input_chunks.json left untouched)")