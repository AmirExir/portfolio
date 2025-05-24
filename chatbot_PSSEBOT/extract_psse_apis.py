from bs4 import BeautifulSoup
import os

input_dir = "path_to_html_docs/_as_gen"  # adjust this path
output_dir = "psse_api_chunks"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".html"):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            title = soup.find("h1").text.strip() if soup.find("h1") else filename
            description = soup.find("p").text.strip() if soup.find("p") else ""
            code_block = soup.find("pre").text.strip() if soup.find("pre") else ""

        with open(os.path.join(output_dir, f"{title}.txt"), "w", encoding="utf-8") as out:
            out.write(f"{title}\n\n{description}\n\n{code_block}")