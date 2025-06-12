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

def count_tokens(text, model="gpt-4o"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def run_executor(prompt, context, valid_funcs):
    model = "gpt-4o"
    
    context_block = f"""
    You are a Python expert in power system automation using the PSS®E API.

    🔧 When performing power system operations, use ONLY the following validated `psspy` functions:
    {', '.join(sorted(valid_funcs))}

    🛠️ For general-purpose tasks like file I/O, plotting, GUI creation, and data analysis, you may use these **common standard libraries**:
    - `os`, `sys`, `re`, `csv`, `json`, `math`, `time`
    - `tkinter` (for GUI)
    - `matplotlib.pyplot` (for plotting)
    - `seaborn` (for advanced plotting)
    - `pandas` (for dataframes and CSV handling)
    - `numpy` (for array math)
    - `sklearn` / `scikit-learn` (only for very basic ML tasks if requested)
    - `streamlit` (for web GUI, optional)

    ⚠️ **Strict Rule**: 
    - DO NOT make up or guess any `psspy` function. Use only those listed above.
    - If you are not sure how to proceed using valid functions, state that clearly in the response.
    - You may safely use the allowed standard libraries for support tasks (e.g., plotting or UI).

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
        max_tokens=max_response_tokens
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