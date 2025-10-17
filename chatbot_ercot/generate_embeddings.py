import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import APIError
from openai._exceptions import APIConnectionError, RateLimitError
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

print(f"Total chunks to embed: {len(chunks)}")

# Retry logic
def get_embedding_with_retry(text, retries=5, delay=5):
    for attempt in range(retries):
        try:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text
            )
            return response.data[0].embedding
        except APIConnectionError as e:
            print(f"Connection error on attempt {attempt + 1}: {e}")
        except RateLimitError as e:
            print(f"Rate limit error on attempt {attempt + 1}: {e}")
        except APIError as e:
            print(f"OpenAI API error on attempt {attempt + 1}: {e}")
            break
        time.sleep(delay * (2 ** attempt))
    return None

# Generate embeddings
embeddings = []
for i, chunk in enumerate(chunks):
    print(f"Embedding chunk {i+1}/{len(chunks)} ({len(chunk['text'].split())} tokens)...")
    embedding = get_embedding_with_retry(chunk["text"])
    if embedding:
        embeddings.append(embedding)
    else:
        print(f" Failed to embed chunk {i+1}. Skipping.")

# Save to disk
save_embeddings(chunks, embeddings, output_json, output_embeddings)
print(f" Saved {len(embeddings)} embeddings to disk.")