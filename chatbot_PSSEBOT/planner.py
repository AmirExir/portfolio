from openai import OpenAI
import os
import tiktoken

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def plan_tasks(user_query, reference_chunks, model="gpt-4o", token_limit=128000, max_response_tokens=32768):
    """
    Plan executable steps to solve the user query using reference PSS/E examples.
    Token-safe version that prevents OpenAI BadRequestError.
    """
    encoding = tiktoken.encoding_for_model(model)

    # Initial prompt headers
    preamble = """
You are a task planner agent specialized in Python automation for PSS/E (power system simulator).
Your job is to break down the user’s task into specific, executable Python steps using only real API functions from the provided documentation context.

Strict Rules:
- ONLY generate tasks related to what the user is asking.
- DO NOT include unrelated areas like GIC, harmonics, dynamics, unless the user explicitly asks.
- Use only functions that appear in the documentation context. No made-up methods.
- Keep task steps clean and short. Use plain English action verbs.
""".strip()

    base_tokens = count_tokens(preamble + user_query, model)

    # Leave enough room for output
    available_context_tokens = token_limit - max_response_tokens - base_tokens
    selected_chunks = []
    total_tokens = 0

    for chunk in reference_chunks:
        chunk_tokens = count_tokens(chunk["text"], model)
        if total_tokens + chunk_tokens > available_context_tokens:
            break
        selected_chunks.append(chunk)
        total_tokens += chunk_tokens

    # Final context
    context_block = "\n\n---\n\n".join(chunk["text"] for chunk in selected_chunks)

    messages = [
        {"role": "system", "content": f"{preamble}\n\nDocumentation context:\n---\n{context_block}\n---"},
        {"role": "user", "content": user_query}
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_response_tokens,
        temperature=0.2,
    )

    return response.choices[0].message.content