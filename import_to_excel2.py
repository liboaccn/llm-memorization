
import json
import string


# {
# "idiom":"for crying out loud",
# "idiom_len":4,
# "match":"Y",
# "predict":"for crying out loud",
# "prompt":"for crying out",
# "last_space":"loud",
# "last_space_len":4,
# "last_word_predict":"loud",
# "idioms_pos":"ADV",
# "predict_pos":"ADV",
# "mean_p_max":0.9461263418197632,
# "mean_h_mean":-0.019867032335582735,
# "mean_h_variance":3.7934913346834924
# }
        # 'mean_p_max': mean_p_max,
        # 'mean_h_mean': mean_h_mean,
        # 'mean_h_variance': mean_h_variance

file_path = './predict_output_2.jsonl'
with open(file_path, 'r') as f:
    for i, line in enumerate(f):
        # print(line)
        data = json.loads(line)
        parent_mean_p_max = data['parent']['mean_p_max']
        parent_mean_h_mean = data['parent']['mean_h_mean'] 
        parent_mean_h_variance = data['parent']['mean_h_variance'] 
        
        child_mean_p_max = data['child']['mean_p_max']
        child_mean_h_mean = data['child']['mean_h_mean'] 
        child_mean_h_variance = data['child']['mean_h_variance'] 

        std_output = str(parent_mean_p_max ) + "|" + \
                     str(parent_mean_h_mean ) + "|" + \
                     str(parent_mean_h_variance)  + "|" + \
                     str(child_mean_p_max) + "|" + \
                     str(child_mean_h_mean)  + "|" + \
                     str(child_mean_h_variance)

        print(std_output)

        # idiom = data['idiom']
        # match = data['match']
        # predict = data['predict']
        # prompt = data['prompt']
        # last_space = data['last_space']
        # last_word_predict = data['last_word_predict']
        # idiom_len =  data['idiom_len'] 
        # last_word_len =  data['last_space_len']
        # probabilities_max_value = data['mean_p_max'] 
        # # probabilities_variance_value = data['probabilities_variance_value']
        # hidden_states_mean_value = data['mean_h_mean'] 
        # hidden_states_variance_value = data['mean_h_variance']
        # idioms_last_word_pos  = data['idioms_pos']
        # predict_last_word_pos = data['predict_pos'] 

        # str_output =idiom + "|"+prompt+ "|"+ last_space + "|" +predict +"|"+ \
        #     str(idiom_len) + "|" + str(last_word_len) +"|"+ \
        #     str(idioms_last_word_pos) + "|" + str(predict_last_word_pos)+"|"+ \
        #     str(match) + "|"+ \
        #     str(probabilities_max_value) + "|" + \
        #     str(hidden_states_mean_value)+"|"+     \
        #     str(hidden_states_variance_value)
        # print(str_output)
        