import torch
import csv
import json
import statistics
from transformers import LlamaForCausalLM
from transformers.models.llama.tokenization_llama import LlamaTokenizer
import torch.nn.functional as F
import numpy as np


def load_llama(model_name_or_path):
    # global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    # max_memory = {k: '32GB' for k in global_devices}
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=True, device_map='cuda',
                                             torch_dtype=torch.float32
                                             )
    return model, tokenizer


def csv_to_json(csv_file_path):
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        json_data = [row for row in reader]
        return json_data


def find_subtensor_indices_torch(A, B):
    for start in range(A.size(0) - B.size(0) + 1):
        # print(start, B.size(0), A[start: start + B.size(0)], B)
        if torch.equal(A[start: start + B.size(0)], B):
            return start, start + B.size(0)
    return -1, -1


def exact_match(generated, gold):
    if gold in generated:
        return True  # matched
    else:
        return False # un-matched


def calculate_mean(data):
    total_prob = []
    total_hidden = []

    for entry in data:
        total_prob.append(entry['prob'])
        total_hidden.append(entry['last_hidden'])

    total_hidden = np.array(total_hidden)

    mean_prob = sum(total_prob) / len(total_prob)
    mean_hidden = np.mean(total_hidden, axis=0).tolist()

    return mean_prob, mean_hidden


def predict_next_token(model, tokenizer, prompt=None, input_ids=None, context_name=None, new_tokens=[]):
    if input_ids is None:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)  # 1,6
    if context_name is not None:
        context_name_ids = tokenizer(context_name, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        start, end = find_subtensor_indices_torch(input_ids[0], context_name_ids[0])
        if start == -1:
            context_name = None
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True,
                        output_hidden_states=True, return_dict=True)
        logits = outputs.logits
        last_logits = logits[:, -1, :]  # torch.Size([32000])
        probabilities = F.softmax(last_logits, dim=-1)
        next_token_id = torch.argmax(probabilities, dim=-1, keepdim=True)
        prob = max(probabilities.tolist()[0])

        if context_name is not None:
            context_hidden = torch.mean(outputs.hidden_states[-1][:, start: end, :], dim=1, keepdim=False)
        last_hidden = outputs.hidden_states[-1][:, -1, :]  # torch.Size([1, 6, 4096])

        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        data = {
            'last_hidden': last_hidden.tolist()[0],
            'prob': prob,
            'context_name_hidden': context_hidden.tolist()[0] if context_name is not None else None
        }
        new_tokens.append(data)

        generated_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return generated_text, input_ids, new_tokens


def predict_parent(r_file, w_file, FEW_SHOT_PROMPT):
    fw = open(w_file, 'w')

    json_data = csv_to_json(r_file)
    count = 0
    for item in json_data:
        count += 1
        print(count)
        child = item.get('child')
        parent = item.get('parent')
        parent_type = item.get('parent_type')

        prompt = "\nQ: Who is {}'s {}?\n".format(child, parent_type)

        max_gen_tokens = 8
        input_ids = None
        child_name = child
        new_tokens =[]

        for _ in range(max_gen_tokens):
            generated_text, input_ids, new_tokens = predict_next_token(model, tokenizer, FEW_SHOT_PROMPT+prompt, input_ids, child_name, new_tokens)
            child_name = None
            generated_text = generated_text.split('\n')[-1].split('A: ')[-1]
            print(generated_text, ", parent:", parent)
            matched = exact_match(generated_text, gold=parent)
            if matched:
                mean_prob, mean_hidden = calculate_mean(new_tokens)
                data = {
                    'iter': _,
                    'match': 'Y' if matched else 'N',
                    'prompt': prompt,

                    'child': child,
                    'parent_type': parent_type,
                    'parent': parent,
                    'generated_parent': generated_text,

                    'context_child_hidden': new_tokens[0]['context_name_hidden'],  # context_parent_hidden
                    'gen_parent_prob': mean_prob,  # the prob of generated token
                    'gen_parent_hidden': mean_hidden,  # gen_parent_hidden
                }
                json_data = json.dumps(data)
                fw.write(json_data + '\n')
                fw.flush()
                break


def predict_child(r_file, w_file, FEW_SHOT_PROMPT):
    fw = open(w_file, 'w')
    with open(r_file, 'r') as f:
        for i, line in enumerate(f):

            data = json.loads(line)
            child = data['child']
            parent = data['parent']
            parent_type = data['parent_type']

            context_child_hidden = data['context_child_hidden']
            gen_parent_prob = data['gen_parent_prob']
            gen_parent_hidden = data['gen_parent_hidden']

            prompt = "\nQ: Name a child of {}?\n".format(parent)

            max_gen_tokens = 4
            input_ids = None
            new_tokens = []
            parent_name = parent
            context_parent_hidden = None
            for _ in range(max_gen_tokens):
                generated_text, input_ids, new_tokens = predict_next_token(model, tokenizer, FEW_SHOT_PROMPT+prompt, input_ids,
                                                                           context_name=parent_name, new_tokens=new_tokens)
                if _ == 0:
                    context_parent_hidden = new_tokens[0]['context_name_hidden']
                parent_name = None
                generated_text = generated_text.split('\n')[-1].split('A: ')[-1]
                matched = exact_match(generated_text, gold=child)
                if matched:
                    break
            mean_prob, mean_hidden = calculate_mean(new_tokens)
            if not matched:
                data = {
                    'iter': _+1,
                    'match': 'Y' if matched else 'N',
                    'prompt': prompt,

                    'child': child,
                    'parent_type': parent_type,
                    'parent': parent,
                    'generated_child': generated_text,

                    'context_child_hidden': context_child_hidden,
                    'gen_parent_prob': gen_parent_prob,  # memorized
                    'gen_parent_hidden': gen_parent_hidden,  # memorized

                    'context_parent_hidden': context_parent_hidden,
                    'gen_child_prob': mean_prob,  # non-memorized
                    'gen_child_hidden': mean_hidden  # non-memorized

                }
                json_data = json.dumps(data)
                fw.write(json_data + '\n')
                fw.flush()


if __name__ == "__main__":
    PROMPT_v1 = """Below is a converation with a helpful and terse assistant. 
    The assistant has knowledge of a wide range of people and can identify people that the user asks for. 
    If the answer is unknown or not applicable, the assistant answers with "I don't know."
    Q: Name a child of Barack Obama.
    A: Malia Obama
    Q: Who is Elon Musk's mother?
    A: Maye Musk
    Q: Who is Kathy Pratt's mother?
    A: I don't know.
    Q: Who is Chris Hemsworth's father?
    A: Craig Hemsworth
    Q: Name a child of Karen Lawrence.
    A: Jennifer Lawrence
    Q: Who is Aaron Taylor-Johnson's mother?
    A: Sarah Johnson"""

    PROMPT_v2 = """
    Q: Name a child of Barack Obama.
    A: Malia Obama
    Q: Who is Elon Musk's mother?
    A: Maye Musk
    Q: Name a child of Donald Trump.
    A: Ivanka Trump
    Q: Who is Chris Hemsworth's father?
    A: Craig Hemsworth
    Q: Name a child of Karen Lawrence.
    A: Jennifer Lawrence
    Q: Who is Aaron Taylor-Johnson's mother?
    A: Sarah Johnson"""

    PROMPT_v3 = "Below is a converation with a helpful and terse assistant. " \
                "The assistant has knowledge of a wide range of people and can identify people that the user asks for. " \
                "If the answer is unknown or not applicable, the assistant answers with 'I don't know.'"

    model_name_or_path = "../llama2-7b-hf"
    model = model_name_or_path.split('/')[-1]
    model, tokenizer = load_llama(model_name_or_path)
    csv_file_path = '../parent_child_pairs.csv'

    # step 1: get the representation of context_child_name, generated_parent_name, context_parent_name
    predict_parent(r_file=csv_file_path,
                   w_file='CelebrityParent_predict_parents_{}_v1.json'.format(model),
                   FEW_SHOT_PROMPT=PROMPT_v1)
    predict_child(r_file='CelebrityParent_predict_parents_{}_v1.json'.format(model),
                  w_file='CelebrityParent_predict_child_{}_v1.json'.format(model),
                  FEW_SHOT_PROMPT=PROMPT_v1)

    predict_parent(r_file=csv_file_path,
                   w_file='CelebrityParent_predict_parents_{}_v2.json'.format(model),
                   FEW_SHOT_PROMPT=PROMPT_v2)
    predict_child(r_file='CelebrityParent_predict_parents_{}_v2.json'.format(model),
                  w_file='CelebrityParent_predict_child_{}_v2.json'.format(model),
                  FEW_SHOT_PROMPT=PROMPT_v2)

    predict_parent(r_file=csv_file_path,
                   w_file='CelebrityParent_predict_parents_{}_v3.json'.format(model),
                   FEW_SHOT_PROMPT=PROMPT_v3)
    predict_child(r_file='CelebrityParent_predict_parents_{}_v3.json'.format(model),
                  w_file='CelebrityParent_predict_child_{}_v3.json'.format(model),
                  FEW_SHOT_PROMPT=PROMPT_v3)



