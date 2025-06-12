from openai import OpenAI
import os
import tiktoken  # make sure to add this to requirements.txt

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def plan_tasks(user_query, reference_chunks, model="gpt-4o"):
    """
    Plan executable steps to solve the user query using reference PSS/E examples.
    Token-safe version that dynamically adjusts max response tokens.
    """
    # Limit context before even formatting
    context_texts = [chunk["text"] for chunk in reference_chunks]
    
    encoding = tiktoken.encoding_for_model(model)
    system_prefix = "You are a task planner agent specialized in Python automation for PSS/E..."
    total_tokens = count_tokens(system_prefix, model=model) + count_tokens(user_query, model=model)
    context_limit = 128000 - total_tokens - 32768  # leave room for response

    selected_texts = []
    for text in context_texts:
        t_count = len(encoding.encode(text))
        if total_tokens + t_count > context_limit:
            break
        selected_texts.append(text)
        total_tokens += t_count

    chunk_context = "\n\n---\n\n".join(selected_texts)

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
""".strip()
    }

    user_prompt = {"role": "user", "content": user_query.strip()}

    input_tokens = sum(count_tokens(m["content"], model=model) for m in [system_prompt, user_prompt])
    max_output_tokens = min(32768, 128000 - input_tokens)

    response = client.chat.completions.create(
        model=model,
        messages=[system_prompt, user_prompt],
        max_tokens=max_output_tokens,
        temperature=0.2,
    )

    return response.choices[0].message.content