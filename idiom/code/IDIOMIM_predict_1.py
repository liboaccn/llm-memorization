"""
step1: request and download llama
step2: convert the downloaded llama checkpoint to Huggingface Transformers format, by conversion script
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
step3: load tokenizer,
see doc: https://huggingface.co/docs/transformers/v4.33.3/en/model_doc/llama
"""

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
    if pre == ground:
        return 'Y'
    else:
        return 'N'


def get_last_word_pos(sentence):
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
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)  # 1,6
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


def generate_idiom(r_file, w_file, model, tokenizer):
    fw = open(w_file, 'w')
    with open(r_file, 'r') as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            s = data['idiom']
            idioms_pos = get_last_word_pos(s)
            idiom_word_num = len(s.split())
            before_last_space = s.rsplit(' ', 1)[0]
            last_space = s.rsplit(' ', 1)[1]

            prompt = before_last_space
            max_gen_tokens = 5  # 5
            input_ids = None
            new_tokens = []
            for i in range(max_gen_tokens):
                generated_text, input_ids, new_tokens = predict_next_token(model, tokenizer, prompt, input_ids,
                                                                           new_tokens)
                generated_text = remove_punctuation(generated_text)
                if len(generated_text.split()) == idiom_word_num + 1:
                    generated_text = ' '.join(generated_text.split()[:-1])
                    predicted_word = generated_text.split()[-1]
                    match = string_match(predicted_word, last_space)

                    predict_pos = get_last_word_pos(generated_text)

                    mean_prob, mean_hidden = calculate_mean(new_tokens)
                    data = {
                        'match': match,
                        'idiom': s,
                        'idiom_len': idiom_word_num,

                        'generated_text': generated_text,
                        'predicted_word': predicted_word,
                        'predicted_word_pos': predict_pos,

                        'last_word': last_space,
                        'last_word_len': len(last_space),
                        'last_word_pos': idioms_pos,

                        'prompt': prompt,

                        'mean_prob': mean_prob,
                        'mean_hidden': mean_hidden,
                    }
                    json_data = json.dumps(data)
                    fw.write(json_data + '\n')
                    break
                elif i == max_gen_tokens - 1:  # 已经生成最大token了，还是没有生成完整idiom
                    generated_text = ' '.join(generated_text.split()[:-1])
                    predicted_word = generated_text.split()[-1]
                    match = string_match(predicted_word, last_space)
                    logging.info('match:{}'.format(match))

                    predict_pos = get_last_word_pos(generated_text)

                    mean_prob, mean_hidden = calculate_mean(new_tokens)
                    data = {
                        'match': match,
                        'idiom': s,
                        'idiom_len': idiom_word_num,

                        'generated_text': generated_text,
                        'predicted_word': predicted_word,
                        'predicted_word_pos': predict_pos,

                        'last_word': last_space,
                        'last_word_len': len(last_space),
                        'last_word_pos': idioms_pos,

                        'prompt': prompt,

                        'mean_prob': mean_prob,
                        'mean_hidden': mean_hidden,
                    }
                    json_data = json.dumps(data)
                    fw.write(json_data + '\n')


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from load_LLMs import load_model, MODELS

    r_file = '../data/idiomem.jsonl'

    for model_name_or_path in MODELS:
        logging.info('Loading model: {}'.format(model_name_or_path))

        w_file = '../data/idiom_out_{}.jsonl'.format(model_name_or_path.split('/')[-1])
        logging.info('written file: {}'.format(w_file))

        model, tokenizer = load_model(model_name_or_path)
        generate_idiom(r_file, w_file, model, tokenizer)
