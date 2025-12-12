from openai import OpenAI
import os
from utils import count_tokens  # <-- reuse shared tokenizer

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def plan_tasks(
    user_query,
    reference_chunks,
    model="gpt-5.2",
    token_limit=120000,
    max_response_tokens=12000
):
    # ---------------------------
    # SYSTEM PROMPT
    # ---------------------------
    preamble = """
You are a task planner agent specialized in Python automation for PSS/E (power system simulator).
Your job is to break down the user’s task into specific, executable Python steps using only real API functions from the provided documentation context.

Strict Rules:
- ONLY generate tasks related to what the user is asking.
- DO NOT include unrelated areas like GIC, harmonics, dynamics, unless the user explicitly asks.
- Use only functions that appear in the documentation context. No made-up methods.
- Keep task steps clean and short. Use plain English action verbs.
""".strip()

    # ---------------------------
    # Token budgeting (MODEL-AGNOSTIC)
    # ---------------------------
    base_tokens = count_tokens(preamble + user_query)
    available_for_chunks = token_limit - max_response_tokens - base_tokens

    if available_for_chunks <= 0:
        return "[Planner Error] Token budget too small."

    selected_chunks = []
    used_tokens = 0

    for chunk in reference_chunks:
        chunk_tokens = count_tokens(chunk["text"])
        if used_tokens + chunk_tokens > available_for_chunks:
            break
        selected_chunks.append(chunk)
        used_tokens += chunk_tokens

    context_block = "\n\n---\n\n".join(chunk["text"] for chunk in selected_chunks)

    messages = [
        {
            "role": "system",
            "content": f"{preamble}\n\nDocumentation context:\n---\n{context_block}\n---",
        },
        {"role": "user", "content": user_query},
    ]

    total_input_tokens = sum(count_tokens(m["content"]) for m in messages)
    print(
        f"[Planner] Tokens: system+user={base_tokens}, "
        f"context={used_tokens}, total={total_input_tokens}"
    )

    try:
        response = client.responses.create(
            model=model,
            reasoning={"effort": "high"},  # correct for planning
            input=messages,
            max_output_tokens=max_response_tokens,
        )

        return response.output_text

    except Exception as e:
        return f"[Planner Error] {str(e)}"