from openai import OpenAI
import os
import tiktoken  # Make sure this is in requirements.txt

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def plan_tasks(user_query, reference_chunks, model="gpt-4o"):
    """
    Plan executable steps to solve the user query using reference PSS/E examples.
    Token-safe version that dynamically adjusts max response tokens.
    """
    chunk_context = "\n\n---\n\n".join(chunk["text"] for chunk in reference_chunks)

    system_prompt_text = f"""
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
""".strip()

    user_prompt_text = user_query.strip()

    messages = [
        {"role": "system", "content": system_prompt_text},
        {"role": "user", "content": user_prompt_text}
    ]

    input_tokens = sum(count_tokens(m["content"], model=model) for m in messages)
    max_available = 128000 - input_tokens
    max_response_tokens = min(max_available, 32768)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_response_tokens,
        temperature=0.2,
    )

    return response.choices[0].message.content