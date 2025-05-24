from bs4 import BeautifulSoup
import os

input_dir = "chatbot_PSSEBOT"  # Folder with all your .html files
output_dir = "psse_api_chunks"  # Output folder for .txt files
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".html"):
        with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            
            # Title fallback
            title_tag = soup.find("title")
            title = title_tag.text.strip().split("–")[0] if title_tag else filename.replace(".html", "")
            
            # Extract h1, h2, pre, and p content
            body_div = soup.find("div", class_="body")
            if body_div:
                parts = []
                for tag in body_div.find_all(["h1", "h2", "h3", "pre", "p"]):
                    parts.append(tag.text.strip())
                content = "\n\n".join(parts)
            else:
                content = "No body content found."

            # Write to file
            with open(os.path.join(output_dir, f"{title}.txt"), "w", encoding="utf-8") as out:
                out.write(f"{title}\n\n{content}")