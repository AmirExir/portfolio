import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from utils import split_text_into_chunks, save_embeddings

# Load API key from .env
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# File paths
input_files = [
    "ercot_planning_part1.txt",
    "ercot_planning_part2.txt",
    "ercot_planning_part3.txt",
]
output_json = "ercot_planning_chunks.json"
output_embeddings = "ercot_planning_embeddings.npy"

# Read and split text
chunks = []
for file in input_files:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        file_chunks = split_text_into_chunks(text, source=file)
        chunks.extend(file_chunks)

# Generate embeddings
print(f"Generating embeddings for {len(chunks)} chunks...")
embeddings = []
for i, chunk in enumerate(chunks):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=chunk["text"]
    )
    embeddings.append(response.data[0].embedding)

# Save to disk
save_embeddings(chunks, embeddings, output_json, output_embeddings)
print(f"Saved {len(embeddings)} embeddings to disk.")