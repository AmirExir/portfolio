import os
import pickle
import numpy as np
from openai import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

# ✅ Load OpenAI key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Path to FAISS + pickle
INDEX_DIR = "ercot_combined_index"
FAISS_INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
PKL_METADATA_PATH = os.path.join(INDEX_DIR, "index.pkl")

# ✅ Load FAISS index
with open(PKL_METADATA_PATH, "rb") as f:
    stored_data = pickle.load(f)

embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.getenv("OPENAI_API_KEY"))
faiss_index = FAISS.load_local(INDEX_DIR, embeddings=embedding_model, index_name="index")

# ✅ Query input
query = input("🔍 Enter your ERCOT question: ")

# ✅ Search top 5 matches
results = faiss_index.similarity_search(query, k=5)

print("\n📌 Top 5 relevant chunks:\n" + "-"*60)
for i, doc in enumerate(results, 1):
    print(f"\n#{i} — Source: {doc.metadata.get('source', 'unknown')} | Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
    print(doc.page_content[:1000])  # limit output
    print("-"*60)