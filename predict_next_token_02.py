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
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import torch.nn.functional as F
import statistics
# import pdb
import spacy

nlp = spacy.load("en_core_web_sm")

def get_last_word_pos(sentence):
    doc = nlp(sentence)
    last_word_pos = doc[-1].pos_
    return last_word_pos


def calculate_mean(data):
    total_p_max = 0
    total_h_mean = 0
    total_h_variance = 0

    for entry in data:
        total_p_max += entry['p_max']
        total_h_mean += entry['h_mean']
        total_h_variance += entry['h_variance']

    num_entries = len(data)
    mean_p_max = total_p_max / num_entries
    mean_h_mean = total_h_mean / num_entries
    mean_h_variance = total_h_variance / num_entries

    return mean_p_max, mean_h_mean, mean_h_variance

def load_llama(model_name_or_path):
    global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    max_memory = {k: '32GB' for k in global_devices}
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=True, device_map='balanced',
                                             torch_dtype=torch.float32, max_memory=max_memory
                                             )
    return model, tokenizer


def predict_next_token(model, tokenizer, prompt=None, input_ids=None, new_tokens=[]):
    if input_ids is None:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device) # 1,6
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
        last_attentions = outputs.attentions[len(outputs.attentions)-1] # torch.Size([32000])
        # print("last_attentions",  last_attentions.shape)   
        #print('hidden', outputs.hidden_states)
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # torch.Size([1, 6, 4096])
        # print("last_hidden", last_hidden) 
        # print("last_hidden", last_hidden.shape) 
        probabilities = F.softmax(last_logits, dim=-1)
        # p = torch.max(probabilities, dim=-1, keepdim=True)
        # print('probabilities',p)
        p_max = max(probabilities.tolist()[0])
        h_mean = statistics.mean(last_hidden.tolist()[0])
        h_variance = statistics.variance(last_hidden.tolist()[0])
        
        
        # p2 = max(probabilities.tolist()[0])
        # print('p2', p2)
        # data['probabilities_mean_value'] = statistics.mean(data['probabilities'])
        # data['probabilities_variance_value'] = statistics.variance(data['probabilities'])


        # print('probabilities',probabilities)
        # Option 1: Select the token with the highest probability
        next_token_id = torch.argmax(probabilities, dim=-1, keepdim=True)
        next_token = tokenizer.convert_ids_to_tokens(next_token_id)
        # print('next_token_id', next_token_id)
        # print('next_token', next_token)
        # Option 2: Sample from the distribution (for more diversity in generation)
        # next_token_id = torch.multinomial(probabilities, 1)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        # print("input_ids2", input_ids)
        data = {
            'p_max': p_max,
            'h_mean': h_mean,
            'h_variance': h_variance,
            # 'next_token': next_token,
            # 'next_token_id': next_token_id.tolist()[0],
        }
        # new_tokens = torch.cat([new_tokens, data], dim=-1)
        new_tokens.append(data)
        generated_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    # return print(generated_text)
    return generated_text, input_ids, new_tokens
        # still need to: 1. convert id to text, 2. dump input, next token and corresponding logits, 3. hidden states
    # return generated_text, probabilities,last_hidden,last_logits 
    # return generated_text, probabilities, input_ids

def string_match(s, m):
    last_char = s[-1]
    last_word = s.rstrip().rstrip(string.punctuation).rsplit(' ', 1)[1]
    # print('last_word', last_word)
    # print('last_char', last_char)
    if m == last_word:
        return True, 'STOP', last_word
    else:
        if not last_char.isalnum():
            return False, 'STOP', last_word
        else:
            return True, 'GO', last_word
# Example usage
model_name_or_path = "../llama2-7b-hf"  # Replace with the path to your LLaMA model
# model_name_or_path = "/home/users/libo15/data/model/Llama-2-7b-hf" 
model, tokenizer = load_llama(model_name_or_path)

# prompt = "The quick brown fox"  # Replace with your prompt

# prompt = "high as a"
# max_gen_tokens = 5
# input_ids = None
# for _ in range(max_gen_tokens):
#     generated_text, input_ids, last_hidden, probabilities  = predict_next_token(model, tokenizer, prompt, input_ids)
#     input_ids      = input_ids 
#     # print("generated_text",generated_text[0])
#     predict_word =  generated_text[0].replace("\n", "") 
#     ok, go, word= string_match(predict_word, 'kite')
#     if go == 'STOP':
#         break

# if go == 'STOP' and ok == True:
#     ex = 'Y'
# else:
#     ex = 'N'
# print('ex', ex)
# print('predict_word',predict_word)
# # print('input_ids', input_ids.tolist())
# print('hidden_states', last_hidden.tolist()[0][0])
# print('probabilities', probabilities.tolist()[0][0])

# max_gen_tokens = 5
# while True:
#     prompt = input("Enter your text (type 'quit' to exit): ")
#     if prompt.lower() == 'quit':
#         break

#     input_ids = None
#     for _ in range(max_gen_tokens):
#         input_ids = predict_next_token(model, tokenizer, prompt, input_ids)

with open('idiomem.jsonl', 'r') as f:
# with open('idiomem_demo.jsonl', 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        s = data['idiom'] 
        before_last_space = s.rsplit(' ', 1)[0]
        last_space = s.rsplit(' ', 1)[1]
        prompt  = before_last_space 
        # prompt = "high as a"
        max_gen_tokens = 5
        input_ids = None
        new_tokens= []
        for _ in range(max_gen_tokens):
            generated_text, input_ids, new_tokens  = predict_next_token(model, tokenizer, prompt, input_ids, new_tokens)
            input_ids      = input_ids 
            new_tokens     = new_tokens
            # print("generated_text",generated_text[0])
            predict_word =  generated_text[0].replace("\n", "") 
            ok, go, word= string_match(predict_word, last_space)
            if go == 'STOP':
                break
        # print('hidden_states', last_hidden.tolist())
        # print('probabilities', probabilities.tolist())
        if go == 'STOP' and ok == True:
            ex = 'Y'
        else:
            ex = 'N'

        # idiom = data['idiom']
        # match = data['match']
        predict = predict_word.rstrip().rstrip(string.punctuation)
        # prompt = data['prompt']
        # last_space = data['last_space']
        # last_word_predict = data['last_word_predict']

        # print(idiom + "|"+prompt+ "|"+ predict + "|" +last_space + "|" + str(match))
        idioms_pos = get_last_word_pos(s)
        # print("idioms_pos",idioms_pos)
        predict_pos = get_last_word_pos(predict)
        # print("predict_pos",predict_pos)
        idiom_len = len(s.split())
        data['idiom_len'] = idiom_len
        data['last_space_len'] = len(last_space) 

        mean_p_max, mean_h_mean, mean_h_variance = calculate_mean(new_tokens)
        data  = {
            'idiom': s,
            'idiom_len': idiom_len,
            'match': ex,
            'predict': predict_word,
            'prompt': prompt,
            'last_space': last_space,
            'last_space_len': len(last_space),
            'last_word_predict': word,
            'idioms_pos': idioms_pos,
            'predict_pos': predict_pos,
            'mean_p_max': mean_p_max,
            'mean_h_mean': mean_h_mean,
            'mean_h_variance': mean_h_variance
            # 'hidden_states': last_hidden.tolist()[0],
            # 'probabilities': probabilities.tolist()[0]
        }
        json_data = json.dumps(data)
        print(json_data)
        # print(new_tokens)
        # print('=========================================')
        # if i == 20:
        #     break


#  print('ex', ex)
# print('predict_word',predict_word)
# # print('input_ids', input_ids.tolist())
# print('hidden_states', last_hidden.tolist()[0][0])
# print('probabilities', probabilities.tolist()[0][0])       
        # #generated_text = run_inference(model, tokenizer, prompt)
        # generated_text, probabilities,last_hidden,last_logits =predict_next_token(model, tokenizer, prompt, input_ids) 
        # predict_word =  generated_text[0].replace("\n", "") 
        # ex = last_space  in  predict_word
        # print("|"+prompt+ "|"+ predict_word + "|" +last_space + "|" + str(ex))
