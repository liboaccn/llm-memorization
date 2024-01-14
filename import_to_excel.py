
import json
import string

 
file_path = './predict_output.jsonl'
with open(file_path, 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        idiom = data['idiom']
        match = data['match']
        predict = data['predict']
        prompt = data['prompt']
        last_space = data['last_space']
        last_word_predict = data['last_word_predict']
        idiom_len =  data['idiom_len'] 
        last_word_len =  data['last_space_len']
        probabilities_max_value = data['mean_p_max'] 
        # probabilities_variance_value = data['probabilities_variance_value']
        hidden_states_mean_value = data['mean_h_mean'] 
        hidden_states_variance_value = data['mean_h_variance']
        idioms_last_word_pos  = data['idioms_pos']
        predict_last_word_pos = data['predict_pos'] 

        str_output =idiom + "|"+prompt+ "|"+ last_space + "|" +predict +"|"+ \
            str(idiom_len) + "|" + str(last_word_len) +"|"+ \
            str(idioms_last_word_pos) + "|" + str(predict_last_word_pos)+"|"+ \
            str(match) + "|"+ \
            str(probabilities_max_value) + "|" + \
            str(hidden_states_mean_value)+"|"+     \
            str(hidden_states_variance_value)
        print(str_output)
        