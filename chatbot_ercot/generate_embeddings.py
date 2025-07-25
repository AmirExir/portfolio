import openai
import numpy as np
import json

openai.api_key = "YOUR_OPENAI_API_KEY"

# Load JSON file with chunks
with open("ercot_planning_chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

texts = [chunk["text"] for chunk in chunks]

embeddings = []
batch_size = 100

for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    response = openai.Embedding.create(
        model="text-embedding-3-large",
        input=batch
    )
    batch_embeddings = [item["embedding"] for item in response["data"]]
    embeddings.extend(batch_embeddings)

# Save the numpy array
np.save("ercot_planning_embeddings.npy", np.array(embeddings))