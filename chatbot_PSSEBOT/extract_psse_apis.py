from bs4 import BeautifulSoup
import os
import re

# Set paths
input_dir = "chatbot_PSSEBOT"  # Folder with your .html API files
output_dir = "psse_api_extracted"
os.makedirs(output_dir, exist_ok=True)

# Loop through each HTML file
for filename in os.listdir(input_dir):
    if filename.endswith(".html"):
        filepath = os.path.join(input_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")

            # Get title (API name)
            title = soup.find("h1").text.strip() if soup.find("h1") else filename

            # Clean title for use in filenames
            clean_title = re.sub(r'[\\/*?:"<>|()\s]', "_", title)

            # Extract code block (e.g., Python call)
            code_block = soup.find("pre").text.strip() if soup.find("pre") else ""

            # Optional: short description
            description = soup.find("p").text.strip() if soup.find("p") else ""

            # Save to .txt
            output_file = os.path.join(output_dir, f"{clean_title}.txt")
            with open(output_file, "w", encoding="utf-8") as out:
                out.write(f"{title}\n\n{description}\n\n{code_block}")