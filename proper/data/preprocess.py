# c = set()
# with open('proper_noun.txt', encoding='utf-8') as f:
#     for line in f:
#         t = line.strip().split()
#         if len(t) >= 4:
#             c.add(line.strip())
# for x in c:
#     print(x)
import json

fw = open('proper_noun_last_one.jsonl', 'w', encoding='utf-8')
with open('proper_noun.txt', encoding='utf-8') as f:
    for line in f:
        t = line.strip().split()
        data = {}
        data['noun'] = line.strip()
        data['noun_count'] = len(line.strip().split())
        answer = t[-1]
        t[-1] = '_'
        prompt = ' '.join(t)
        data['prompt'] = prompt
        data['answer'] = answer

        fw.write(json.dumps(data, ensure_ascii=False) + '\n')
