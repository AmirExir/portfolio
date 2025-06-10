from openai import OpenAI
import os
import json

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def plan_tasks(user_query, reference_chunks):
    """
    Plan steps to solve the user query using the reference documentation chunks.
    This acts as a basic planner agent.
    """
    chunk_context = "\n\n---\n\n".join(chunk["text"] for chunk in reference_chunks)

    system_prompt = {
        "role": "system",
        "content": f"""
        You are a task planner agent specialized in Python automation for PSS/E (power system simulator).
        Given a user query and supporting documentation, break the problem down into clear, executable tasks
        that a code-generation agent could follow. Output the plan as a list of tasks.

        Documentation context:
        ---
        {chunk_context}
        ---

        Rules:
        - Make each task concise and specific
        - Use only real PSS/E functions and concepts from the context
        - Do not invent API methods or tasks not supported by the documentation
        """
    }

    user_prompt = {"role": "user", "content": user_query}

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[system_prompt, user_prompt],
        max_tokens=2048
    )

    task_plan = response.choices[0].message.content
    return task_plan