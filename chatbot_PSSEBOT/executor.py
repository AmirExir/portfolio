# executor.py
import re
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_valid_funcs(chunks):
    pattern = r'psspy\.(\w+)' 
    valid = set()
    for chunk in chunks:
        valid.update(re.findall(pattern, chunk["text"]))
    return valid

def run_executor(prompt, context, valid_funcs):
    system_prompt = {
        "role": "system",
        "content": f"""
You are a Python expert in power system automation using the PSS/E API.
Use only valid PSSPY functions from the documentation context below.
Do NOT make up any functions. If you're unsure about a function, do not use it.

Documentation Context:
---
{context}
---
        """
    }

    user_prompt = {"role": "user", "content": prompt}

    messages = [system_prompt, user_prompt]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    output = response.choices[0].message.content
    used_funcs = re.findall(r'psspy\.(\w+)', output)
    invalid = [f for f in used_funcs if f not in valid_funcs]

    if invalid:
        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": f"""
You used invalid or hallucinated PSSPY functions: {invalid}.
Please regenerate your response using only valid functions from the context above.
""".strip()})
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        output = response.choices[0].message.content

    return output