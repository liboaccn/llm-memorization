# from random import random
import random
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


def create_prompt_ground(term, ground_truth):
    from load_LLMs import NOUN_PROMPT, NOUN_PROMPT_last_one

    # prompt = NOUN_PROMPT.format(term)
    prompt = NOUN_PROMPT_last_one.format(term)
    max_gen_tokens = 2

    return max_gen_tokens, prompt, term, ground_truth


def generate_noun(r_file, w_file, model, tokenizer):
    fw = open(w_file, 'w', encoding='utf-8')
    with open(r_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = json.loads(line)
            noun = line['prompt']
            ground_truth = line['answer']

            max_gen_tokens, prompt, previous_line, ground_truth = create_prompt_ground(noun, ground_truth)

            input_ids = None
            new_tokens = []
            for i in range(max_gen_tokens):
                generated_text, input_ids, new_tokens = predict_next_token(model, tokenizer, prompt, input_ids,
                                                                           new_tokens)
                generated_text = remove_punctuation(generated_text.replace(prompt, ''))
                print('generated: ---', generated_text)

                match = string_match(generated_text, ground_truth)
                if ground_truth in generated_text:
                    match = 'Y'
                    break
            mean_prob, mean_hidden = calculate_mean(new_tokens)
            line['match'] = match
            line['mean_prob'] = mean_prob
            line['generated_text'] = generated_text
            line['mean_hidden'] = mean_hidden
            line['answer_pos'] = get_word_pos(ground_truth)
            line['answer_len'] = len(ground_truth)

            data = {
                'match': match,

                'answer': line['answer'],
                'generated_text': generated_text,
                'mean_prob': mean_prob,
                'prompt': noun,

                'answer_pos': get_word_pos(ground_truth),
                'answer_len': len(ground_truth),

                'noun_count': line['noun_count'],
                'noun': line['noun'],

                'mean_hidden': mean_hidden,

            }

            json_data = json.dumps(data, ensure_ascii=False)
            fw.write(json_data + '\n')


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from load_LLMs import load_model, MODELS

    # r_file = '../data/proper_noun.jsonl'
    r_file = '../data/proper_noun_last_one.jsonl'

    for model_name_or_path in MODELS:
        logging.info('Loading model: {}'.format(model_name_or_path))

        w_file = '../data/noun_out_last_one_{}.jsonl'.format(model_name_or_path.split('/')[-1])
        logging.info('written file: {}'.format(w_file))

        model, tokenizer = load_model(model_name_or_path)
        generate_noun(r_file, w_file, model, tokenizer)
