import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
import os

# Set your OpenAI key securely (you can also use dotenv or streamlit secrets)
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.getenv("OPENAI_API_KEY")

# Constants
INDEX_DIR = "ercot_index"  # or whatever your FAISS folder name is
INDEX_NAME = "index"

# Load FAISS index
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
faiss_index = FAISS.load_local(INDEX_DIR, embedding_model, index_name=INDEX_NAME, allow_dangerous_deserialization=True)

# Streamlit App
st.title("ERCOT AI Assistant")
st.markdown("Ask me about ERCOT Planning Guide, Protocols, or Interconnection Handbook!")

query = st.text_input("🧠 Enter your question about ERCOT:")

if query:
    results = faiss_index.similarity_search(query, k=5)

    st.subheader("🔍 Top Matches")
    for i, doc in enumerate(results):
        st.markdown(f"**[{i+1}]** `{doc.metadata.get('source', 'unknown')}`")
        st.write(doc.page_content)
        st.markdown("---")