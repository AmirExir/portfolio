from retriever import load_chunks_and_embeddings

# This command validates the deployed saved index. Corpus embedding is an
# explicit offline maintenance operation and is never triggered by app startup.
chunks, embeddings = load_chunks_and_embeddings("input_chunks.json")

print(
    "Saved PSS/E index validated: "
    f"{len(chunks)} chunks, {embeddings.shape[1]} dimensions. "
    "No embedding API requests were made."
)
