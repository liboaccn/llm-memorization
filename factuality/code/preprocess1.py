
import json

# Load the existing JSON data
with open('../data/popQA.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract the entries starting from the 10th one
extracted_data = data[11: 11+1000]  # 10th entry has index 9

# Write the extracted entries to a new JSON Lines file
with open('../data/popQA.jsonl', 'w', encoding='utf-8') as file:
    for entry in extracted_data:
        file.write(json.dumps(entry,  ensure_ascii=False) + '\n')



# Load the existing JSON data
with open('../data/LAMA_UHN.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract the entries starting from the 10th one
extracted_data = data[11: 11+1000]  # 10th entry has index 9

# Write the extracted entries to a new JSON Lines file
with open('../data/LAMA_UHN.jsonl', 'w', encoding='utf-8') as file:
    for entry in extracted_data:
        file.write(json.dumps(entry,  ensure_ascii=False) + '\n')



