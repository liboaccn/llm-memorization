import json

with open('idiom_predict.jsonl', 'r') as f:
    num_list = []
    for i, line in enumerate(f):
        data = json.loads(line)
        num = data['idiom_len']
        num_list.append(int(num))
print(sum(num_list)/len(num_list))