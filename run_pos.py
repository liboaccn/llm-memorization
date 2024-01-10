

# pip install spacy
# python -m spacy download en_core_web_sm


import spacy
import statistics
import json
import string

# 加载英文模型
nlp = spacy.load("en_core_web_sm")

def get_last_word_pos(sentence):
    doc = nlp(sentence)
    last_word_pos = doc[-1].pos_
    return last_word_pos

file_path = './predict_output.jsonl'
with open(file_path, 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        idiom = data['idiom']
        match = data['match']
        predict = data['predict'].rstrip().rstrip(string.punctuation)
        prompt = data['prompt']
        last_space = data['last_space']
        last_word_predict = data['last_word_predict']

        # print(idiom + "|"+prompt+ "|"+ predict + "|" +last_space + "|" + str(match))
        idioms_pos = get_last_word_pos(idiom)
        # print("idioms_pos",idioms_pos)
        predict_pos = get_last_word_pos(predict)
        # print("predict_pos",predict_pos)
        idiom_len = len(idiom.split())
        data['idiom_len'] = idiom_len
        data['last_space_len'] = len(last_space) 
        
        data['probabilities_mean_value'] = statistics.mean(data['probabilities'])
        data['probabilities_variance_value'] = statistics.variance(data['probabilities'])


        data['hidden_states_mean_value'] = statistics.mean(data['hidden_states'])
        data['hidden_states_variance_value'] = statistics.variance(data['hidden_states'])

        data['idioms_pos'] = idioms_pos
        data['predict_pos'] = predict_pos
        json_data = json.dumps(data)
        # put to predict_output_2.jsonl 
        print(json_data)
