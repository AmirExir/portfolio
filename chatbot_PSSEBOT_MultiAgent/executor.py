# executor.py
import re
import os
from openai import OpenAI
from utils import count_tokens

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_valid_funcs(chunks):
    pattern = r'\bpsspy\.(\w+)\b'
    valid = set()
    for chunk in chunks:
        valid.update(re.findall(pattern, chunk["text"]))
    return valid


def run_executor(
    prompt: str,
    context: str,
    valid_funcs: set,
    model: str = "gpt-5.2",
    token_limit: int = 120000,
    max_response_tokens: int = 12000,
):
    """
    Generate valid PSS/E Python code using verified API functions.
    """

    system_prompt = f"""
You are a senior power system automation engineer and PSS®E Python expert.

Your task is to generate **valid, executable Python code** using the PSS®E API (`psspy`).

Strict rules:
- Use ONLY PSSPY functions that appear in the provided documentation context.
- DO NOT invent or guess function names.
- You MAY use standard Python libraries (numpy, pandas, os, csv, glob, time, matplotlib).
- Prefer real examples if present in the documentation.
- If something is unclear, choose the safest documented approach.

Documentation context:
---
{context}
---
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]

    # ---------------------------
    # Token budgeting (model-agnostic)
    # ---------------------------
    input_tokens = sum(count_tokens(m["content"]) for m in messages)
    available = token_limit - input_tokens
    max_out = min(max_response_tokens, available)

    if max_out <= 0:
        return "[Executor Error] Token budget exceeded."

    # ---------------------------
    # First generation
    # ---------------------------
    response = client.responses.create(
        model=model,
        reasoning={"effort": "xhigh"},  # executor deserves xhigh
        input=messages,
        max_output_tokens=max_out,
    )

    output = response.output_text

    # ---------------------------
    # Validate PSSPY usage
    # ---------------------------
    used_funcs = re.findall(r'\bpsspy\.(\w+)\b', output)
    invalid = sorted(set(f for f in used_funcs if f not in valid_funcs))

    if not invalid:
        return output

    # ---------------------------
    # Self-correction loop
    # ---------------------------
    correction_prompt = f"""
The previous answer used invalid or undocumented PSS®E API functions:

{', '.join(invalid)}

Please revise the solution using ONLY valid functions from the documentation context.
Do NOT introduce new API names.
""".strip()

    correction_messages = messages + [
        {"role": "assistant", "content": output},
        {"role": "user", "content": correction_prompt},
    ]

    correction_response = client.responses.create(
        model=model,
        reasoning={"effort": "xhigh"},
        input=correction_messages,
        max_output_tokens=max_out,
    )

    corrected_output = correction_response.output_text

    return (
        "⚠️ **Auto-correction applied due to invalid API usage**\n\n"
        + corrected_output
    )