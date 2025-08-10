from openai import OpenAI
import os
import tiktoken

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def count_tokens(text, model="gpt-5o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def plan_tasks(user_query, reference_chunks, model="gpt-5", token_limit=120000, max_response_tokens=12000):
    encoding = tiktoken.encoding_for_model(model)

    # SYSTEM PROMPT
    preamble = """
You are a task planner agent specialized in Python automation for PSS/E (power system simulator).
Your job is to break down the user’s task into specific, executable Python steps using only real API functions from the provided documentation context.

Strict Rules:
- ONLY generate tasks related to what the user is asking.
- DO NOT include unrelated areas like GIC, harmonics, dynamics, unless the user explicitly asks.
- Use only functions that appear in the documentation context. No made-up methods.
- Keep task steps clean and short. Use plain English action verbs.
""".strip()

    # Calculate base token usage (system + user)
    base_tokens = count_tokens(preamble + user_query, model)
    available_for_chunks = token_limit - max_response_tokens - base_tokens

    # Select chunks that fit in remaining budget
    selected_chunks = []
    total_chunk_tokens = 0
    for chunk in reference_chunks:
        chunk_tokens = count_tokens(chunk["text"], model)
        if total_chunk_tokens + chunk_tokens > available_for_chunks:
            break
        selected_chunks.append(chunk)
        total_chunk_tokens += chunk_tokens

    # Format context
    context_block = "\n\n---\n\n".join(chunk["text"] for chunk in selected_chunks)

    # Final messages
    messages = [
        {"role": "system", "content": f"{preamble}\n\nDocumentation context:\n---\n{context_block}\n---"},
        {"role": "user", "content": user_query}
    ]

    # Log the token use
    total_input_tokens = sum(count_tokens(m["content"], model) for m in messages)
    print(f"[Planner] Tokens: system+user={base_tokens}, context={total_chunk_tokens}, total={total_input_tokens}")

    # Call API with safe margins
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_response_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"[Planner] ❌ OpenAI API Error: {e}")
        return f"[Error] Could not generate plan due to: {str(e)}"