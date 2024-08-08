import torch
import json
import string
import torch.nn.functional as F
import statistics
# import pdb
import spacy
import numpy as np

nlp = spacy.load("en_core_web_sm")


def remove_punctuation(input_string):
    # 定义要去除的标点符号
    punctuations = string.punctuation

    # 创建一个映射表，将所有标点符号映射为None
    translator = str.maketrans('', '', punctuations)

    # 使用translate方法去除标点符号
    return input_string.translate(translator)


def string_match(pre, ground):
    if ground in pre or pre in ground:
        return 'Y'
    else:
        return 'N'


def get_word_pos(sentence):
    doc = nlp(sentence)
    last_word_pos = doc[-1].pos_
    return last_word_pos


def calculate_mean(data):
    total_prob = []
    total_hidden = []

    for entry in data:
        total_prob.append(entry['prob'])
        total_hidden.append(entry['hidden'])

    total_hidden = np.array(total_hidden)

    mean_prob = sum(total_prob) / len(total_prob)
    mean_hidden = np.mean(total_hidden, axis=0).tolist()

    return mean_prob, mean_hidden


def predict_next_token(model, tokenizer, prompt=None, input_ids=None, new_tokens=[]):
    if input_ids is None:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)  # 1,15
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True,
                        output_hidden_states=True, return_dict=True)

        # get the predicted token, and the probability of the predicted token
        logits = outputs.logits  # torch.Size([1, 6, 32000])
        last_logits = logits[:, -1, :]  # torch.Size([32000])
        probabilities = F.softmax(last_logits, dim=-1)

        next_token_id = torch.argmax(probabilities, dim=-1, keepdim=True)
        prob = max(probabilities.tolist()[0])

        # get hidden states of the predicted token
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # torch.Size([1, 6, 4096])

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        data = {
            'prob': prob,
            'hidden': last_hidden.tolist()[0],
        }
        new_tokens.append(data)
        generated_text = \
            tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return generated_text, input_ids, new_tokens


def create_prompt_ground(question, answer, prompt_num=6):
    from load_LLMs import PROMPT_10

    prompt = PROMPT_10.format(question)

    # 上下句
    ground_truth = answer
    max_gen_tokens = 3

    return max_gen_tokens, prompt, ground_truth


def generate(r_file, w_file, model, tokenizer, prompt_num=6):
    fw = open(w_file, 'w', encoding='utf-8')
    with open(r_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            question = line['question']
            answer = line['answer']

            max_gen_tokens, prompt, ground_truth = create_prompt_ground(question, answer, prompt_num)

            input_ids = None
            new_tokens = []
            match = 'N'
            for i in range(max_gen_tokens):
                generated_text, input_ids, new_tokens = predict_next_token(model, tokenizer, prompt, input_ids,
                                                                           new_tokens)
                generated_text = remove_punctuation(generated_text.replace(prompt, ''))
                print('generated: ---', generated_text)

                match = string_match(generated_text, ground_truth)
                if ground_truth in generated_text:
                    match = 'Y'
                    break
            print('\n')
            mean_prob, mean_hidden = calculate_mean(new_tokens)
            data = {
                'match': match,
                'ground_truth': ground_truth,
                'generated_text': generated_text,

                'question': question,
                'question_len': len(question.split()),
                'answer_len': len(answer),
                'answer_pos': get_word_pos(answer),

                'mean_prob': mean_prob,
                'mean_hidden': mean_hidden,
            }

            json_data = json.dumps(data, ensure_ascii=False)
            fw.write(json_data + '\n')


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from load_LLMs import load_model, MODELS

    r_file = '../data/popQA.jsonl'

    for prompt_num in [10]:
        for model_name_or_path in MODELS:
            logging.info('Loading model: {}'.format(model_name_or_path))

            w_file = '../data/popQA_out_{}_shot_{}.jsonl'.format(prompt_num, model_name_or_path.split('/')[-1])
            logging.info('written file: {}'.format(w_file))

            model, tokenizer = load_model(model_name_or_path)
            generate(r_file, w_file, model, tokenizer, prompt_num)
