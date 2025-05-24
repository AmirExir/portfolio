from bs4 import BeautifulSoup
import os

input_dir = r"C:\Program Files\PTI\PSS\E35\35.6\DOCS\Sphinx\psspy\_as_gen"
output_dir = "psse_api_extracted"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(".html"):
        filepath = os.path.join(input_dir, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")
            # Find the main API block
            api_block = soup.find("dl", class_="py function")
            if api_block:
                dt = api_block.find("dt")
                dd = api_block.find("dd")
                if dt and dd:
                    title = dt.get_text(strip=True)
                    desc = dd.get_text(separator="\n", strip=True)
                    # Save as text
                    output_file = os.path.join(output_dir, f"{title}.txt")
                    with open(output_file, "w", encoding="utf-8") as out:
                        out.write(f"{title}\n\n{desc}")