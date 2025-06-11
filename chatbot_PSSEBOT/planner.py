# planner.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def plan_tasks(user_query, reference_chunks):
    """
    Plan executable steps to solve the user query using reference PSS/E examples.
    Reject hallucinations and only include verified PSS/E API tasks.
    """
    chunk_context = "\n\n---\n\n".join(chunk["text"] for chunk in reference_chunks)

    system_prompt = {
        "role": "system",
        "content": f"""
    You are a task planner agent specialized in Python automation for PSS/E (power system simulator).

    Your job is to break down the user’s task into specific, executable Python steps using only real API functions from the provided documentation context.

    Documentation context:
    ---
    {chunk_context}
    ---

    Strict Rules:
    - ONLY generate tasks related to what the user is asking.
    - DO NOT include unrelated areas like GIC, harmonics, dynamics, unless the user explicitly asks.
    - Use only functions that appear in the documentation context. No made-up methods.
    - Keep task steps clean and short. Use plain English action verbs.
    """
    }

    user_prompt = {"role": "user", "content": user_query}

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_prompt, user_prompt],
        max_tokens=2048,
        temperature=0.2,
    )

    task_plan = response.choices[0].message.content
    return task_plan