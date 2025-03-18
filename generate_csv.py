import json
import pandas as pd

# Path to your annotation file
json_file = 'annotation.json'

# Load the JSON data
with open(json_file, 'r') as f:
    data = json.load(f)

# Create separate CSV files for each split (train and test)
for split in ["train", "test"]:
    rows = []
    # Assuming your JSON structure is something like: { "train": [...], "test": [...] }
    for entry in data.get(split, []):
        case_id = entry.get('id', '')
        report = entry.get('report', '')
        image_paths = entry.get('image_path', [])
        if image_paths:
            for img in image_paths:
                rows.append({
                    'id': case_id,
                    'report': report,
                    'image_path': img,
                    'split': split
                })
    df = pd.DataFrame(rows)
    csv_filename = f"data/{split}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"{csv_filename} created with {len(df)} samples.")
