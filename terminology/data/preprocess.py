import json

fw = open('desease_terminology.jsonl', 'w', encoding='utf-8')

with open('common_desease.txt', encoding='utf-8') as f:
    for line in f:
        save_line = {}
        if len(line.split()) > 2:
            print(line.strip())
            t = line.strip().split()
            answer = t[-2]
            t[-2] = '_'
            prompt = ' '.join(t)

            save_line['type'] = 'common_desease'
            save_line['terminology'] = line.strip()
            save_line['term_count'] = len(line.strip().split())

            save_line['prompt'] = prompt
            save_line['answer'] = answer

            json_data = json.dumps(save_line, ensure_ascii=False)
            fw.write(json_data + '\n')

with open('rare_desease.txt', encoding='utf-8') as f:
    for line in f:
        save_line = {}
        if len(line.split()) > 2:
            print(line.strip())
            t = line.strip().split()
            answer = t[-2]
            t[-2] = '_'
            prompt = ' '.join(t)

            save_line['type'] = 'rare_desease'
            save_line['terminology'] = line.strip()
            save_line['term_count'] = len(line.strip().split())

            save_line['prompt'] = prompt
            save_line['answer'] = answer

            json_data = json.dumps(save_line, ensure_ascii=False)
            fw.write(json_data + '\n')