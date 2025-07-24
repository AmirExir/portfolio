import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI

# Set OpenAI API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set the path to the FAISS index directory
INDEX_DIR = "ercot_combined_index"
PKL_METADATA_PATH = os.path.join(INDEX_DIR, "index.pkl")

# Load metadata stored with the FAISS index
with open(PKL_METADATA_PATH, "rb") as f:
    stored_data = pickle.load(f)

# Load the embedding model
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Load the FAISS index with embedded chunks and allow pickle deserialization
faiss_index = FAISS.load_local(
    INDEX_DIR,
    embeddings=embedding_model,
    index_name="index",
    allow_dangerous_deserialization=True
)

# Prompt user for a query
query = input("🧠 Enter your ERCOT question: ")

# Search the index (top 5 matches)
results = faiss_index.similarity_search(query, k=5)

# Display results
print("\n🔎 Top 5 results:\n" + "="*60)
for i, doc in enumerate(results, 1):
    print(f"\n[{i}] 📄 Source: {doc.metadata.get('source', 'unknown')} | 🧩 Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
    print(doc.page_content[:1000])  # Print up to 1000 characters
    print("-"*60)