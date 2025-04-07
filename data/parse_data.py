import pandas as pd
from datasets import load_dataset

def parse_bible_text(file_path):
    bible_dict = {}

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # First try tab-separated format (English Genesis)
            if '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    ref, text = parts
                    bible_dict['en ' + ref.strip()] = text.strip()
                    continue
            
            # For space-separated formats (Hebrew and Greek)
            # Find the reference pattern (book name + chapter:verse)
            parts = line.split(' ')
            
            # Look for the chapter:verse pattern to determine where reference ends
            ref_end_idx = -1
            for i, part in enumerate(parts):
                if ':' in part:
                    ref_end_idx = i
                    break
            
            if ref_end_idx >= 0:
                # Combine book name (might have spaces) with chapter:verse
                ref_parts = parts[:ref_end_idx+1]
                reference = ' '.join(ref_parts)
                
                # The rest is the text
                text = ' '.join(parts[ref_end_idx+1:])
                
                # Clean up any special markers like ¶
                text = text.replace('¶', '').strip()
                
                bible_dict[reference] = text
                
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return bible_dict


# Example usage
english_file_path = "data/berean_bible.txt"  # Path to your Bible text file
greek_file_path = "data/tr.txt"
hebrew_file_path = "data/he_modern.txt"
bible_dict = {}
bible_dict.update(parse_bible_text(english_file_path))
bible_dict.update(parse_bible_text(greek_file_path))
bible_dict.update(parse_bible_text(hebrew_file_path))


data = [{"reference": ref, "text": text} for ref, text in bible_dict.items()]
df = pd.DataFrame(data)

# Display the first few rows
print(df.tail())

df.to_json("data/bible_dataset.json", orient="records", indent=4, force_ascii=False)

# Load the dataset from the JSON file
dataset = load_dataset("json", data_files="data/bible_dataset.json")

dataset.save_to_disk("data/hugging_face_bible_dataset_format")

# Print a sample
print(dataset['train'][0])
