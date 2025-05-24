from bs4 import BeautifulSoup
import os

input_dir = r"C:\Program Files\PTI\PSS\E35\35.6\DOCS\Sphinx\psspy\_as_gen"  # update if needed
output_dir = "psse_api_extracted"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".html"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            title = soup.find("h1")
            description = soup.find("dd")
            if title and description:
                title_text = title.get_text(strip=True)
                desc_text = description.get_text(strip=True)
                output_path = os.path.join(output_dir, f"{title_text}.txt")
                with open(output_path, "w", encoding="utf-8") as out:
                    out.write(f"{title_text}\n\n{desc_text}")