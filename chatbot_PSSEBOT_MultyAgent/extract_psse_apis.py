import os
import json
from bs4 import BeautifulSoup

api_dir = "path/to/_as_gen"  # the folder with psspy_xxx.html files
output = []

for filename in os.listdir(api_dir):
    if not filename.endswith(".html"):
        continue

    path = os.path.join(api_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Extract function name
    title = soup.find("h1")
    if not title or "psspy" not in title.text:
        continue

    func_name = title.text.strip()

    # Try to find description
    desc_tag = soup.find("div", {"class": "description"})
    description = desc_tag.text.strip() if desc_tag else "No description available."

    # Optionally grab code example
    code_block = soup.find("pre")
    code = code_block.text.strip() if code_block else ""

    output.append({
        "function": func_name,
        "description": description,
        "code": code
    })

# Save as JSON
with open("psse_api_functions.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"✅ Extracted {len(output)} functions to psse_api_functions.json")