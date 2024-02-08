import torch
import csv
import json
import string
import statistics
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm

# import pdb
def calculate_mean(data):
    num_entries = len(data)
    if num_entries == 0:
        return 0, 0, 0
    total_p_max = 0
    total_h_mean = 0
    total_h_variance = 0

    for entry in data:
        total_p_max += entry['p_max']
        total_h_mean += entry['h_mean']
        total_h_variance += entry['h_variance']

    # num_entries = len(data)
    mean_p_max = total_p_max / num_entries
    mean_h_mean = total_h_mean / num_entries
    mean_h_variance = total_h_variance / num_entries

    return mean_p_max, mean_h_mean, mean_h_variance




root_path = os.path.dirname(os.path.abspath(__file__))+'/'
def save_jsonl_file(fname, data):
    fpath = root_path + '/save/'+fname+'.jsonl'
    # print(fpath)
    folder_path = os.path.dirname(fpath)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(fpath, 'a') as f:
        f.write(data + "\n")
        print('Save data file :'+ fpath)

# Function to find the start and end indices of B in A
def find_subtensor_indices(A, B):
    start = -1
    end = -1
    len_B = len(B)
    
    # Iterate through A to find a match
    for i in range(len(A) - len_B + 1):
        if A[i:i+len_B] == B:
            start = i
            end = i + len_B - 1
            break  # Assuming B appears only once in A, we stop after finding the first match
    
    return start, end

def load_llama(model_name_or_path):
    global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    max_memory = {k: '32GB' for k in global_devices}
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, legacy=False)
    # model = LlamaForCausalLM.from_pretrained(model_name_or_path,
    #                                          low_cpu_mem_usage=True, device_map='balanced',
    #                                          torch_dtype=torch.float32, max_memory=max_memory
    #                                          )
    model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=True, device_map='cuda',
                                             torch_dtype=torch.float32
                                             )

    return model, tokenizer


def predict_next_token(model, tokenizer, prompt=None, input_ids=None, new_tokens=[], name = None):
    print('prompt', prompt)
    print('name', name)
    if input_ids is None:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device) # 1,6
    name_ids  = tokenizer(name, return_tensors="pt").input_ids.to(model.device) # 1,6
    # print("input_ids", input_ids.tolist()[0][1:])
    # print("name_ids", name_ids.tolist()[0][1:])
    # print("input_ids1", input_ids)
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True,
                            # max_length=max_length,
                             output_hidden_states=True, return_dict=True)
        # print("logits",  outputs.logits.shape)  # torch.Size([1, 6, 32000]) B*L*V
        # for i in range(len(outputs.attentions)): # 32
        #     print("attentions", outputs.attentions[i].shape) # torch.Size([1, 32, 6, 6])
        # for i in range(len(outputs.hidden_states)): # 33
        #     print("hidden_states",  outputs.hidden_states[i].shape)  # torch.Size([1, 6, 4096]

        # pdb.set_trace()

        logits = outputs.logits # torch.Size([1, 6, 32000])
        # print("logits",  logits)
        # Get logits for the last token
        last_logits = logits[:, -1, :] # torch.Size([32000])
        # print("last_logits",  last_logits.shape)  
        
        #print('attentions', outputs.attentions)
        # last_attentions = outputs.attentions[len(outputs.attentions)-1] # torch.Size([32000])
        # print("last_attentions",  last_attentions.shape)   
        hidden = outputs.hidden_states[-1]
        name_s, name_e = find_subtensor_indices(input_ids.tolist()[0][1:], name_ids.tolist()[0][1:])
        name_hidden = hidden[:, name_s:name_e, :]
        # print('name_hidden', name_hidden.tolist())
        #print('hidden', outputs.hidden_states)
        last_hidden = hidden[:, -1, :]  # torch.Size([1, 6, 4096])
        # print("last_hidden", last_hidden) 
        # print("last_hidden", last_hidden.shape) 
        probabilities = F.softmax(last_logits, dim=-1)
        # p_max = max(probabilities.tolist()[0])
        # h_mean = statistics.mean(last_hidden.tolist()[0])
        # h_variance = statistics.variance(last_hidden.tolist()[0])
       
        # print('probabilities',probabilities)
        # Option 1: Select the token with the highest probability
        next_token_id = torch.argmax(probabilities, dim=-1, keepdim=True)
        next_token = str(tokenizer.convert_ids_to_tokens(next_token_id)[0])
        print('next_token', next_token)
        # Option 2: Sample from the distribution (for more diversity in generation)
        # next_token_id = torch.multinomial(probabilities, 1)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        data = {
            # 'p_max': p_max,
            # 'h_mean': h_mean,
            # 'h_variance': h_variance,
            'next_token': next_token,
            'next_token_id': next_token_id.tolist()[0],
            'last_hidden': last_hidden.tolist()[0],
            'name_hidden': name_hidden.tolist()[0],
        }
        # print(data)
        last_char = next_token[-1]
        # new_tokens = torch.cat([new_tokens, data], dim=-1)
        if last_char.isalnum()  or last_char.isspace() or last_char == '-' or last_char == "'":
            new_tokens.append(data)
        else:
            new_tokens = []
        # print('new_tokens', new_tokens)
        # print("input_ids2", input_ids)
        generated_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print('gt', generated_text)
    return generated_text, input_ids, new_tokens#last_hidden, probabilities 
        # still need to: 1. convert id to text, 2. dump input, next token and corresponding logits, 3. hidden states
    # return generated_text, probabilities,last_hidden,last_logits 
    # return generated_text, probabilities, input_ids

def string_match(s, m):
    last_char = s[-1]
    # last_word = s.rstrip().rstrip(string.punctuation).rsplit(' ', 1)[1]
    # print('last_word', last_word)
    # print('last_char', last_char)
    if m in s:
        return True, 'STOP', s
    else:
        if (not last_char.isalnum() ) and ( not last_char.isspace() ) and ( not last_char == '-') and ( not last_char == "'"):
            return False, 'STOP', s
        else:
            return True, 'GO', s
# Example usage
# model_name_or_path = "../llama2-7b-hf"  # Replace with the path to your LLaMA model
model_name_or_path = "/home/users/libo15/data/model/Llama-2-7b-hf" 
model, tokenizer = load_llama(model_name_or_path)



def csv_to_json(csv_file_path):
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        json_data = [row for row in reader]
        return json_data
        
        


csv_file_path='parent_child_pairs.csv'
# csv_file_path='demo.csv'
json_data = csv_to_json(csv_file_path)
# print(json_data)
# prefix_promptA = "Brooklyn's father is David Beckham. Jaden's mother is Jada Pinkett Smith. So "
prefix_promptA = "Print the names of parents and children of celebrities. Brooklyn's father is David Beckham. Jaden's mother is Jada Pinkett Smith. So "
# prefix_promptA = " "

prefix_promptB = "Print the names of parents and children of celebrities. David Beckham's child is Brooklyn. Will Smith's children are Jaden and Willow. So "

pbar = tqdm(total=len(json_data), desc='Predicting')

for item in json_data:
    pbar.update(1)
    # print(item)
    # break
    print("*"*50)
    child = item.get('child')
    parent_type = item.get('parent_type')
    parent = item.get('parent')
    # print(f'Child: {child}, Parent Type: {parent_type}, Parent: {parent}')
    
    promptA = (f'{child}\'s {parent_type} is')
    promptB = (f'{parent}\'s child is')

    max_gen_tokens = 5
    input_ids = None
    new_tokens =[]
    last_hidden_arr = []
    name_hidden_arr = []
    predict_next = ""
    i = 0
    for _ in range(max_gen_tokens):
        generated_text, _input_ids, _new_tokens = predict_next_token(model, tokenizer, prefix_promptA+promptA, input_ids, new_tokens, child)
        i += 1
        input_ids      = _input_ids 
        new_tokens     = _new_tokens
        if new_tokens:
            # print('nex', new_tokens[-1])
            last_hidden_arr.append(new_tokens[-1]['last_hidden'])
            name_hidden_arr = new_tokens[-1]['name_hidden']
            predict_next  = predict_next + " " + new_tokens[-1]['next_token']

        # last_hidden_arr.append(new_tokens[-1]['last_hidden'])
        # name_hidden_arr = new_tokens[-1]['name_hidden']
        # print("generated_text",generated_text[0])
        predict_word =  generated_text[0].replace("\n", "") 
        ok, go, word= string_match(predict_word, parent)
        if go == 'STOP':
            break
            
    # print('hidden_states', last_hidden.tolist())
    # print('probabilities', probabilities.tolist())
    print('len_last_hidden', len(last_hidden_arr))
    print('i', i)
    last_hidden = np.mean(np.array(last_hidden_arr), axis=0)
    name_hidden = np.mean(np.array(name_hidden_arr), axis=0)
    # print('last_hidden', last_hidden.tolist())
    # print('name_hidden', name_hidden)
    if go == 'STOP' and ok == True:
        ex = 'Y'
    else:
        ex = 'N'
    # mean_p_max, mean_h_mean, mean_h_variance = calculate_mean(new_tokens)
   
    data_a  = {
        # 'idiom': s,
        'match': ex,
        'predict': predict_word,
        'prompt': promptA,
        'last_space': parent,
        'name': child,
        'predict_next': predict_next,
        'last_hidden': last_hidden.tolist(),
        'name_hidden': name_hidden.tolist(),
        # 'last_word_predict': word,
        # 'new_tokens': new_tokens,
        # 'mean_p_max': mean_p_max,
        # 'mean_h_mean': mean_h_mean,
        # 'mean_h_variance': mean_h_variance 
        # # 'hidden_states': last_hidden.tolist()[0],
        # 'probabilities': probabilities.tolist()[0]
    }
    # json_data1 = json.dumps(data)
    # print(data_a)
    # break
    if ex == 'N':
        continue
    # put to predict_output.jsonl 
    # print(json_data1)
    print("-"*50)
    #=========================================================================
    input_ids = None
    new_tokens = []
    last_hidden_arr = []
    name_hidden_arr = []
    predict_next = ""
    j = 0
    for _ in range(max_gen_tokens):
        generated_text, _input_ids, _new_tokens  = predict_next_token(model, tokenizer, prefix_promptB+promptB, input_ids, new_tokens, parent)
        j += 1
        input_ids      = _input_ids 
        new_tokens     = _new_tokens
        if new_tokens:
            # print('nex', new_tokens[-1])
            # print("generated_text",generated_text[0])
            print('*****NEXT:',new_tokens[-1]['next_token'])
            predict_next  = predict_next + " " + new_tokens[-1]['next_token']
            last_hidden_arr.append(new_tokens[-1]['last_hidden'])
            name_hidden_arr = new_tokens[-1]['name_hidden']
        predict_word =  generated_text[0].replace("\n", "") 
        ok, go, word= string_match(predict_word, child)
        if go == 'STOP':
            break
    # print('hidden_states', last_hidden.tolist())
    # print('probabilities', probabilities.tolist())
    print('len_last_hidden', len(last_hidden_arr))
    print('j', j)
    last_hidden = np.mean(np.array(last_hidden_arr), axis=0)
    name_hidden = np.mean(np.array(name_hidden_arr), axis=0)
    if go == 'STOP' and ok == True:
        ex = 'Y'
    else:
        ex = 'N'
    
    # mean_p_max, mean_h_mean, mean_h_variance = calculate_mean(new_tokens)

    data_b  = {
        # 'idiom': s,
        'match': ex,
        'predict': predict_word,
        'prompt': promptB,
        'predict_next': predict_next,
        'last_space': child,
        # 'last_word_predict': word,
        # 'new_tokens': new_tokens,
        'last_hidden': last_hidden.tolist(),
        'name': parent,
        'name_hidden': name_hidden.tolist(),
        # 'mean_p_max': mean_p_max,
        # 'mean_h_mean': mean_h_mean,
        # 'mean_h_variance': mean_h_variance
        # 'hidden_states': last_hidden.tolist()[0],
        # 'probabilities': probabilities.tolist()[0]
    }
    # print(data_b)
    # json_data2 = json.dumps(data)
    output = {
        "parent": data_a,
        "child": data_b
    }
    # put to predict_output.jsonl 
    # if ex == 'N':
    # print(output)
    # print(json.dumps(output,indent='\t'))
    save_jsonl_file('predict_rep_child_and_parent0228', json.dumps(output))
    # print('===================================================================')
    # break
pbar.close()