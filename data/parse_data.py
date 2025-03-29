import pandas as pd
from datasets import load_dataset

def parse_bible_text(file_path):
    bible_dict = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Skip the metadata and headers
    parsing = False

    for line in lines:
        line = line.strip()

        # Start parsing after the header
        if line.startswith("Verse") or line.startswith("Genesis 1:1"):
            parsing = True
            continue
        
        if not parsing or not line:
            continue

        # Split by the first tab or multiple spaces, handling both formats
        parts = line.split('\t', 1)
        
        if len(parts) != 2:
            continue  # Skip lines that don't match the format

        ref, text = parts
        ref = ref.strip()
        text = text.strip()

        # Add to dictionary
        bible_dict[ref] = text

    return bible_dict


# Example usage
file_path = "data/berean_bible.txt"  # Path to your Bible text file
bible_dict = parse_bible_text(file_path)


data = [{"reference": ref, "text": text} for ref, text in bible_dict.items()]
df = pd.DataFrame(data)

# Display the first few rows
print(df.head())

df.to_json("data/bible_dataset.json", orient="records", indent=4)

# Load the dataset from the JSON file
dataset = load_dataset("json", data_files="bible_dataset.json")

dataset.save_to_disk("data/hugging_face_bible_dataset_format")

# Print a sample
print(dataset['train'][0])
