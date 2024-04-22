import json

previous_line = []
next_line = []

with open('../data/唐诗三百首.txt', encoding='utf-8') as f:
    for line in f:
        line = line.replace("！", "。").replace("？", "。")
        line = line.split("。")[0].split('，')
        if len(line) == 2 and (len(line[0]) == len(line[1])):
            print(line)
            previous_line.append(line[0])
            next_line.append(line[1])

poems_list = []
for previous_line, next_line in zip(previous_line, next_line):
    poems = {}

    poems['id'] = len(poems_list) + 1
    poems['previous_line'] = previous_line
    poems['next_line'] = next_line
    poems_list.append(poems)

with open('../data/tangshi.jsonl', 'a', encoding='utf-8') as f:
    for poems in poems_list:
        f.write(json.dumps(poems, ensure_ascii=False) + '\n')