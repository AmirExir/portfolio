# 3. executor.py
import re

def extract_valid_funcs(chunks):
    pattern = r'psspy\.(\w+)' 
    valid = set()
    for chunk in chunks:
        valid.update(re.findall(pattern, chunk["text"]))
    return valid

def run_executor(prompt, context, valid_funcs):
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = [
        {"role": "system", "content": f"""
            You are a PSS/E automation expert. Use only real API from this reference:
            ---\n{context}\n---
            No made-up functions. Provide working Python.
        """},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    output = response.choices[0].message.content

    used_funcs = re.findall(r'psspy\.(\w+)', output)
    invalid = [f for f in used_funcs if f not in valid_funcs]

    if invalid:
        messages.append({"role": "assistant", "content": output})
        messages.append({"role": "user", "content": f"You used invalid functions: {invalid}. Retry with only valid ones."})
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
        output = response.choices[0].message.content

    return output
