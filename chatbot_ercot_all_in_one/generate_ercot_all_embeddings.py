import os
import json

source_dir = "ercot_sources"
output_file = "ercot_combined_chunks.json"

# How many characters per chunk
CHUNK_SIZE = 800

chunks = []
for filename in os.listdir(source_dir):
    if filename.endswith(".txt"):
        filepath = os.path.join(source_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            source = filename.replace(".txt", "")
            for i in range(0, len(text), CHUNK_SIZE):
                chunk_text = text[i:i+CHUNK_SIZE].strip()
                if chunk_text:
                    chunks.append({
                        "text": chunk_text,
                        "source": source,
                        "chunk_index": i // CHUNK_SIZE
                    })

# Save to disk
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"✅ Chunked {len(chunks)} sections into {output_file}")