import os
import time
import json
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
from utils import split_text_into_chunks, save_embeddings

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in .env file.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# File paths
input_files = [
    "ercot_planning_part1.txt",
    "ercot_planning_part2.txt",
    "ercot_planning_part3.txt",
]
output_json = "ercot_planning_chunks.json"
output_embeddings = "ercot_planning_embeddings.npy"

# Tokenizer for safety
encoding = tiktoken.encoding_for_model("text-embedding-3-large")
max_tokens = 8191  # model limit

# Read and split text
chunks = []
for file in input_files:
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
        file_chunks = split_text_into_chunks(text, source=file)
        chunks.extend(file_chunks)

print(f"Total chunks to embed: {len(chunks)}")

# Generate embeddings
embeddings = []
for i, chunk in enumerate(chunks):
    text = chunk["text"]
    num_tokens = len(encoding.encode(text))
    if num_tokens > max_tokens:
        print(f"Chunk {i} too long ({num_tokens} tokens), skipping.")
        continue

    print(f"Embedding chunk {i + 1}/{len(chunks)} ({num_tokens} tokens)...")

    for attempt in range(3):  # Retry logic
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            embeddings.append(response.data[0].embedding)
            break
        except Exception as e:
            print(f"Error on chunk {i}: {e}")
            if attempt < 2:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print("Failed after 3 attempts. Skipping.")
                embeddings.append([0.0] * 3072)  # Filler vector to maintain length match

# Save results
save_embeddings(chunks, embeddings, output_json, output_embeddings)
print(f"✅ Saved {len(embeddings)} embeddings to disk.")