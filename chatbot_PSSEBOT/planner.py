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
You are a task planner specialized in Python automation for PSS®E (Power System Simulator for Engineering).
Given a user request and relevant documentation context, break down the query into specific tasks.

📌 Rules:
- Use only real PSS/E APIs or functions from the documentation below.
- Do NOT invent or guess PSSPY methods. Every task must be grounded in the provided context.
- Format output as a clear step-by-step task list.
- If the query is unanswerable based on the context, say so.

Documentation:
---
{chunk_context}
---
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