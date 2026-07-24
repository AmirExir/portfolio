import streamlit as st
import re
from planner import plan_tasks
from retriever import load_chunks_and_embeddings, find_relevant_chunks
from executor import extract_valid_funcs, run_executor
from utils import parse_planner_tasks

st.set_page_config(page_title="Amir Exir's PSS/E Agent Loop")
st.title("Amir Exir's PSS/E Automation Agent")

MAX_TASKS = 12

if "stop_execution" not in st.session_state:
    st.session_state.stop_execution = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initial load
try:
    with st.spinner("Loading saved PSS/E documentation index..."):
        chunks, embeddings = load_chunks_and_embeddings()
except Exception as exc:
    st.error(f"PSS/E documentation index could not be loaded: {exc}")
    st.stop()

valid_funcs = extract_valid_funcs(chunks)

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Prompt input
prompt = st.chat_input("Ask a PSS/E automation task...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.stop_execution = False
    query_cache = {}

    # Step 1: Plan
    with st.spinner(" Planning tasks..."):
        try:
            planning_chunks = find_relevant_chunks(
                prompt,
                chunks,
                embeddings,
                k=10,
                max_tokens=12_000,
                query_cache=query_cache,
            )
            planning_error = ""
        except Exception as exc:
            planning_chunks = []
            planning_error = f"{type(exc).__name__}: {exc}"
        if planning_error:
            raw_tasks = f"[Planner Error] retrieval failed ({planning_error})"
        elif not planning_chunks:
            raw_tasks = (
                "[Planner Error] no matching PSS/E documentation was retrieved"
            )
        else:
            raw_tasks = plan_tasks(prompt, planning_chunks)
        task_list = parse_planner_tasks(raw_tasks, max_tasks=MAX_TASKS)

    with st.expander("Planned tasks", expanded=False):
        st.code(raw_tasks)

    if st.button(" Stop Execution"):
        st.session_state.stop_execution = True

    all_results = []

    if not task_list:
        st.error(
            "The planning step did not produce a usable task list, so the "
            "executor was not called. Please retry or make the request more specific."
        )

    for task_number, task in enumerate(task_list, start=1):
        if st.session_state.stop_execution:
            st.warning("Execution manually stopped.")
            break

        st.markdown(f"Executing task {task_number} of {len(task_list)}: `{task}`")

        try:
            relevant_chunks = find_relevant_chunks(
                task,
                chunks,
                embeddings,
                k=10,
                max_tokens=12_000,
                query_cache=query_cache,
            )
            task_retrieval_error = ""
        except Exception as exc:
            relevant_chunks = []
            task_retrieval_error = f"{type(exc).__name__}: {exc}"
        if task_retrieval_error:
            result = f"[Executor Error] retrieval failed ({task_retrieval_error})"
        elif not relevant_chunks:
            result = (
                "[Executor Error] no matching documentation was retrieved for "
                f"this task: {task}"
            )
        else:
            combined_context = "\n\n---\n\n".join(
                f"[{chunk.get('id', 'reference')}]\n{chunk['text']}"
                for chunk in relevant_chunks
            )

            with st.spinner("Generating documented Python code..."):
                result = run_executor(task, combined_context, valid_funcs)

        used_funcs = re.findall(r'psspy\.(\w+)', result)
        invalid_funcs = sorted(set(f for f in used_funcs if f not in valid_funcs))

        if invalid_funcs:
            st.warning(
                "The answer still contains undocumented psspy functions: "
                f"{', '.join(invalid_funcs)}. It was not regenerated again to "
                "avoid a duplicate paid request."
            )

        all_results.append(f"### {task}\n\n{result}")

    # Step 4: Final Summary Output
    if all_results:
        final_output = "\n\n".join(all_results)
        st.chat_message("assistant").markdown(final_output)

        st.download_button(
            label="Download output as .txt",
            data=final_output,
            file_name="psse_automation_output.txt",
            mime="text/plain",
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_output
        })

    if st.button("Reset agent"):
        st.session_state.stop_execution = False
        st.session_state.messages = []
        st.rerun()
