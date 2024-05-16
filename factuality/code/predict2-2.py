
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
    if ground in pre:
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
    # from load_LLMs import PROMPT_10

    if prompt_num == 10:
        prompt = PROMPT_10.format(question)
    else:
        prompt = question

    # 上下句
    ground_truth = answer
    max_gen_tokens = 4

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
                generated_text = remove_punctuation(generated_text).replace(prompt, '')
                match = string_match(generated_text, ground_truth)
                if match == "Y":
                    break
            
            mean_prob, mean_hidden = calculate_mean(new_tokens)
            data = {
                'match': match,
                'question': question,
                'answer': answer,
                'question_len': len(question.split()),
                'answer_len': len(answer.split()),
                'answer_pos': get_word_pos(answer),

                'ground_truth': ground_truth,
                'generated_text': generated_text.split()[0],
                'generated_pos': get_word_pos(generated_text),

                'mean_prob': mean_prob,
                'mean_hidden': mean_hidden,
            }

            print("original: ", question, answer)
            print("Generated: ", generated_text, '\n')

            

            json_data = json.dumps(data, ensure_ascii=False)
            fw.write(json_data + '\n')


# 默认
PROMPT_10 = """
question: Paul Mounsey (born 15 April 1959) is a composer arranger and producer from [MASK].
answer: Scotland

question: Ze'ev Jabotinsky MBE (Hebrew: זאב ז'בוטינסקי; born Vladimir Yevgenyevich Zhabotinsky Russian: Влади́мир Евге́ньевич Жаботи́нский; 18 October 1880 Odessa – 4 August 1940 New York City) was a Russian Jewish Revisionist Zionist leader author poet orator soldier and founder of the Jewish Self-Defense Organization in [MASK].
answer: Odessa

question: Pierre Dupont (April 23 1821 – July 25 1870) French song-writer the son of a blacksmith was born in [MASK].
answer: Lyon

question: Susette La Flesche (later Susette LaFlesche Tibbles) also called Inshata Theumba (Bright Eyes) (1854 – 1903) was a well-known Native American writer lecturer interpreter and artist of the Omaha tribe in [MASK].
answer: Nebraska

question: Godert Alexander Gerard Philip Baron van der Capellen (December 15 1778 – April 10 1848) was a Dutch statesman from [MASK].
answer: Utrecht

question: Pietro Andrea Gregorio Mattioli (Matthiolus) ([ˈpjɛːtro anˈdrɛːa ɡreˈɡɔːrjo matˈtjɔːli]; 12 March 1501 – 1577) was a doctor and naturalist born in [MASK].
answer: Siena

question: Corri was born in Rome and studied voice with Nicola Porpora in [MASK].
answer: Naples

question: The Herb Carnegie Centennial Centre formerly named the North York Centennial Centre is a multi-purpose arena located in North York now a part of the city of [MASK].
answer: Toronto

question: Tic Tac is a 1997 Swedish thriller film directed by Daniel Alfredson and written by Hans Renhäll about various people involved in small crime during one day and night in [MASK].
answer: Stockholm

question: Rawlinson reported his meeting to McNeill at Teheran on November 1 and the news soon reached Calcutta and [MASK].
answer: London

question: The revolt of Husayn ibn Ali ibn Hasan broke out when Husayn declared himself caliph in [MASK].
answer: Medina

question: {}
answer:
"""
if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    from load_LLMs import load_model, MODELS

    r_file = '../data/LAMA_UHN.jsonl'

    for prompt_num in [10]:
        for model_name_or_path in MODELS:
            logging.info('Loading model: {}'.format(model_name_or_path))

            w_file = '../data/LAMA_UHN_out_{}_shot_{}.jsonl'.format(prompt_num, model_name_or_path.split('/')[-1])
            logging.info('written file: {}'.format(w_file))

            model, tokenizer = load_model(model_name_or_path)
            generate(r_file, w_file, model, tokenizer, prompt_num)


