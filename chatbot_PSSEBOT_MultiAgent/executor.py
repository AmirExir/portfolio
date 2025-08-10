# executor.py
import re
import os
from openai import OpenAI
import tiktoken  # make sure to install this: pip install tiktoken

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_valid_funcs(chunks):
    pattern = r'psspy\.(\w+)' 
    valid = set()
    for chunk in chunks:
        valid.update(re.findall(pattern, chunk["text"]))
    return valid

def count_tokens(text, model="gpt-5o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def run_executor(prompt, context, valid_funcs):
    model = "gpt-5o"
    
    context_block = f"""
    You are a Python expert in power system automation using the PSS®E API (psspy), and you are allowed to use supporting standard Python libraries when helpful.

    Your job is to generate valid Python code for engineers working with PSS®E.
    Use only verified PSSPY functions provided in the context below, and if needed, use standard Python libraries such as:
    - `tkinter`, `pandas`, `numpy`, `matplotlib`, `os`, `csv`, `glob`, `time`, etc.
    - You may also use simple file handling and GUI operations.

    ⚠️ Strict Rules:
    - Do NOT make up or guess PSSPY functions.
    - Use only what is documented in the provided PSS/E chunks.
    - You MAY use standard Python libraries for GUI, file I/O, or plotting.

    Documentation Context:
    ---
    {context}
    ---
    """.strip()

    messages = [
        {"role": "system", "content": context_block},
        {"role": "user", "content": prompt}
    ]

    # Estimate total token usage and adjust output cap
    input_tokens = sum(count_tokens(m["content"], model) for m in messages)
    max_available = 120000 - input_tokens
    max_response_tokens = min(max_available, 12000)

    # Generate first response
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_response_tokens,
        temperature=0.0,
    )
    output = response.choices[0].message.content

    # Extract and check function calls
    used_funcs = re.findall(r'psspy\.(\w+)', output)
    invalid = [f for f in used_funcs if f not in valid_funcs]

    if invalid:
        flagged_msg = f"""
⚠️ **Warning: The following PSSPY functions are not recognized and may be hallucinated:**

`{', '.join(sorted(set(invalid)))}`

Please double-check the functions against the PSS/E documentation or JSON source. The code below may contain errors.
        """.strip()
        return flagged_msg + "\n\n---\n\n" + output

    return output