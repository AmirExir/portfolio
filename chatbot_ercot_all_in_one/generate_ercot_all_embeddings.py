import os
import json
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

import os
print("🔑 OPENAI_API_KEY is:", os.getenv("OPENAI_API_KEY"))
# Step 1: Load all .txt files from a folder
TEXT_FOLDER = "ercot_sources"  # Change this if your files are in a subfolder
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Step 2: Chunk the content
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

all_documents = []
for filename in os.listdir(TEXT_FOLDER):
    if filename.endswith(".txt"):
        filepath = os.path.join(TEXT_FOLDER, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()
        base_name = os.path.splitext(filename)[0]
        chunks = text_splitter.split_text(raw_text)
        for i, chunk in enumerate(chunks):
            metadata = {"source": base_name, "chunk_id": i}
            all_documents.append(Document(page_content=chunk, metadata=metadata))

# Optional: Save chunks to JSON
json_chunks = [{"content": doc.page_content, "metadata": doc.metadata} for doc in all_documents]
with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(json_chunks, f, indent=2)

# Step 3: Generate and save embeddings
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_documents, embedding_model)
vectorstore.save_local("ercot_combined_index")

# Also save as pickle (optional)
with open("embeddings.pkl", "wb") as f:
    pickle.dump(vectorstore, f)

print(f"✅ {len(all_documents)} chunks embedded and saved!")