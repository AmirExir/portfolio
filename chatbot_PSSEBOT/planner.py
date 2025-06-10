# planner.py
from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def plan_tasks(user_input: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a task planner. Break down the user's PSS/E automation request into step-by-step programming tasks."},
            {"role": "user", "content": user_input}
        ],
        max_tokens=512
    )
    return response.choices[0].message.content.split('\n')