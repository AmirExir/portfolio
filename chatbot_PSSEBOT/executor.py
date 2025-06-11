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
    import re
    from openai import OpenAI
    import os

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Prepare system and user messages
    messages = [
        {
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
        },
        {"role": "user", "content": prompt}
    ]

    # Generate first response
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    output = response.choices[0].message.content

    # Extract function calls
    used_funcs = re.findall(r'psspy\.(\w+)', output)
    invalid = [f for f in used_funcs if f not in valid_funcs]

    # If hallucinated functions are found, flag them instead of retrying
    if invalid:
        flagged_msg = f"""
⚠️ **Warning: The following PSSPY functions are not recognized and may be hallucinated:**

`{', '.join(sorted(set(invalid)))}`

Please double-check the functions against the PSS/E documentation or JSON source. The code below may contain errors.
        """.strip()
        return flagged_msg + "\n\n---\n\n" + output

    # Otherwise, return clean output
    return output