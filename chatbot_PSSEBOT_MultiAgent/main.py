# 4. main.py (streamlit app)
import streamlit as st
from planner import plan_tasks
from retriever import load_chunks_and_embeddings, find_relevant_chunks
from executor import extract_valid_funcs, run_executor

st.title(" Amir Exir's PSS/E Agent Loop")
prompt = st.chat_input("Ask a PSSE task...")

if prompt:
    st.chat_message("user").markdown(prompt)
    tasks = plan_tasks(prompt)
    st.write("**Planned tasks:**", tasks)

    chunks, embeddings = load_chunks_and_embeddings()
    valid_funcs = extract_valid_funcs(chunks)

    all_results = []
    for task in tasks:
        st.markdown(f"### 🔍 Executing: `{task}`")
        top_chunks = find_relevant_chunks(task, chunks, embeddings)
        context = "\n---\n".join(c["text"] for c in top_chunks)
        result = run_executor(task, context, valid_funcs)
        st.markdown(result)
        all_results.append(result)

    st.session_state["chat"] = all_results