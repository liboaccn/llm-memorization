"""
step1: request and download llama
step2: convert the downloaded llama checkpoint to Huggingface Transformers format, by conversion script
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
step3: load tokenizer,
see doc: https://huggingface.co/docs/transformers/v4.33.3/en/model_doc/llama
"""


import torch
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama import LlamaTokenizer, LlamaTokenizerFast
import torch.nn.functional as F


def load_llama(model_name_or_path):
    global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    max_memory = {k: '40GB' for k in global_devices}
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=True, device_map='balanced',
                                             torch_dtype=torch.float32, max_memory=max_memory
                                             )
    return model, tokenizer


def predict_next_token(model, tokenizer, prompt=None, input_ids=None, max_length=5):
    if input_ids is None:
        input_ids = tokenizer.tokenize(prompt, add_special_tokens=False)
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=False,
                             output_hidden_states=True, return_dict=True)
        logits = outputs.logits
        # Get logits for the last token
        last_logits = logits[0, -1, :]
        probabilities = F.softmax(last_logits, dim=-1)

        # Option 1: Select the token with the highest probability
        next_token_id = torch.argmax(probabilities).unsqueeze(0)
        # Option 2: Sample from the distribution (for more diversity in generation)
        next_token_id = torch.multinomial(probabilities, 1)
        input_ids = torch.cat([input_ids, next_token_id], dim=-1)
        hidden_states = outputs.hidden_states  # ???

        # still need to: 1. convert id to text, 2. dump input, next token and corresponding logits, 3. hidden states

    return input_ids

# Example usage
model_name_or_path = "path/to/your/llama/model"  # Replace with the path to your LLaMA model
model, tokenizer = load_llama(model_name_or_path)

prompt = "The quick brown fox"  # Replace with your prompt
max_gen_tokens = 5
input_ids = None
for _ in range(max_gen_tokens):
    input_ids = predict_next_token(model, tokenizer, prompt, input_ids)

