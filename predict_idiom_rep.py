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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import csv
import numpy as np


def load_llama(model_name_or_path):
    global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    max_memory = {k: '32GB' for k in global_devices}
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=True, device_map='balanced',
                                             torch_dtype=torch.float32, max_memory=max_memory
                                             )
    return model, tokenizer


def predict_next_token(model, tokenizer, prompt=None, input_ids=None):
    if input_ids is None:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device) # 1,6
        # print("input_ids1", input_ids)
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True,
                            # max_length=max_length,
                             output_hidden_states=True, return_dict=True)

        rep = outputs.hidden_states[-1][:, :-1, :].tolist()  # torch.Size([1, 6, 4096])
        # print(rep[0][0])
    return rep[0][0]
        
# # Example usage
# #model_name_or_path = "../llama2-7b-hf"  # Replace with the path to your LLaMA model
model_name_or_path = "/home/users/libo15/data/model/Llama-2-7b-hf" 
model, tokenizer = load_llama(model_name_or_path)
    # input_ids = predict_next_token(model, tokenizer, prompt, input_ids)


def plot_tsne(data, label, marker, color, title):

    tsne = TSNE(n_components=2, random_state=42)
    input = np.array(data)
    embedded_hidden_states = tsne.fit_transform(input[:,None],input[:,0])

    plt.scatter(embedded_hidden_states[:, 0], embedded_hidden_states[:, 1], label=label, marker=marker, color=color)

    plt.title(title)
    plt.legend()

    plt.show()


csv_file_path = 'prompt.csv'
y_samples=[]
n_samples=[] 
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    csv_reader = csv.DictReader(csvfile)
    for row in csv_reader:
        # print(row)
        idiom = row['idiom']
        match = row['match']
        rep = predict_next_token(model, tokenizer, idiom)  
        if match == 'Y':
            y_samples.append(rep)
        elif match == 'N':
            n_samples.append(rep)    



# plot_tsne(y_samples, label='Y', marker='^', color='blue', title='t-SNE Visualization of Transformer Hidden States for Y group')
# plot_tsne(n_samples, label='N', marker='o', color='red', title='t-SNE Visualization of Transformer Hidden States for N group')

group_a = np.array(y_samples)
group_b = np.array(n_samples)

# Combine the groups
combined_data = np.vstack((group_a, group_b))

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=10, learning_rate=200, n_iter=1000, random_state=0)
tsne_results = tsne.fit_transform(combined_data)

# Split the results back into two groups
tsne_a = tsne_results[:len(group_a)]
tsne_b = tsne_results[len(group_a):]

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(tsne_a[:, 0], tsne_a[:, 1], color='red', label='Group A')
plt.scatter(tsne_b[:, 0], tsne_b[:, 1], color='blue', label='Group B')
plt.legend()
plt.title('t-SNE visualization of two groups')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()