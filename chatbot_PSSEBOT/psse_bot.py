import streamlit as st
import os
import re
from openai import OpenAI
from planner import plan_tasks
from retriever import load_chunks_and_embeddings, find_relevant_chunks
from executor import extract_valid_funcs, run_executor

# Setup
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
st.set_page_config(page_title="Amir Exir's PSS/E Agent Loop", page_icon="🧠")
st.title("🧠 Amir Exir's PSS/E Automation Agent")

# Initial load
with st.spinner("🤖 Loading PSS/E examples and computing embeddings..."):
    chunks, embeddings = load_chunks_and_embeddings()
st.success(f"✅ Loaded {len(chunks)} chunks from `psse_examples_chunks.json`")

# Prompt input
prompt = st.chat_input("Ask a PSS/E automation task...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages = st.session_state.get("messages", []) + [{"role": "user", "content": prompt}]

    # Step 1: Plan
    with st.spinner("🛠️ Planning tasks..."):
        reference_context = chunks[:30]
        tasks = plan_tasks(prompt, reference_context)
        st.markdown("**🧩 Planned Tasks:**")
        st.code(tasks)

    # Step 2: Extract valid PSSPY functions
    with st.spinner("🔎 Extracting valid API functions..."):
        valid_funcs = extract_valid_funcs(chunks)

    # Step 3: Execute each task
    task_list = [t.strip("- ") for t in tasks.strip().split("\n") if t.strip()]
    st.markdown("---")

    all_results = []
    for task in task_list:
        st.markdown(f"### 🚀 Executing Task: `{task}`")
        relevant_chunks = find_relevant_chunks(task, chunks, embeddings)
        combined_context = "\n---\n".join(chunk["text"] for chunk in relevant_chunks)

        with st.spinner("💻 Generating valid Python code..."):
            result = run_executor(task, combined_context, valid_funcs)
        st.markdown(result)

        # Check for hallucinated functions
        used_funcs = re.findall(r'psspy\.(\w+)', result)
        invalid_funcs = [f for f in used_funcs if f not in valid_funcs]
        if invalid_funcs:
            st.warning(f"⚠️ Invalid functions detected: {invalid_funcs}")
            if st.button(f"🔁 Retry `{task}` with correction", key=task):
                with st.spinner("♻️ Retrying with valid functions..."):
                    result = run_executor(task, combined_context, valid_funcs)
                st.markdown(result)

        all_results.append(result)

    # Step 4: Final Summary Output
    st.markdown("---")
    st.markdown("## 📝 Final Summary")

    full_output = "\n\n".join(all_results)
    st.text_area("🧠 Generated Automation Code", value=full_output, height=400)

    st.download_button(
        label="📥 Download Output as .txt",
        data=full_output,
        file_name="psse_automation_output.txt",
        mime="text/plain"
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_output
    })