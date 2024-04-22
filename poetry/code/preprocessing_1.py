import json


all_3_rows = []
all_4_rows = []


def process_line(line):
    line = line.rstrip().strip()
    line = line.split("。")
    line.pop()
    if len(line) == 2:
        # row_1, row_2 = line[0].split('，')
        line_last = line[1].split('，')
        if len(line_last) == 2:
            row_3, row_4 = line_last
            if len(row_3) == len(row_4) == 7:
                return [row_3, row_4]
    else:
        return None


with open('../data/train.csv', encoding='utf-8') as f:
    for line in f:
        t = process_line(line)
        if t:
            all_3_rows.append(t[0])
            all_4_rows.append(t[1])

# with open('../data/test.csv', encoding='utf-8') as f:
#     for line in f:
#         t = process_line(line)
#         if t:
#             all_3_rows.append(t[0])
#             all_4_rows.append(t[1])

all_3_rows = list(set(all_3_rows))[:2000]
all_4_rows = list(set(all_4_rows))[:2000]

poems_list = []
for row_3, row_4 in zip(all_3_rows, all_4_rows):
    poems = {}

    poems['id'] = len(poems_list) + 1
    poems['previous_line'] = row_3
    poems['next_line'] = row_4
    poems_list.append(poems)

with open('../data/poems.jsonl', 'a', encoding='utf-8') as f:
    for poems in poems_list:
        f.write(json.dumps(poems, ensure_ascii=False) + '\n')