import torch
from transformers import LlamaForCausalLM
from transformers.models.llama import LlamaTokenizer


def load_llama_model(model_name_or_path):
    global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    max_memory = {k: '40GB' for k in global_devices}
    # Load the model and tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=True, device_map='balanced',
                                             torch_dtype=torch.float32, max_memory=max_memory
                                             )
    # Ensure that the model outputs hidden states
    model.config.output_hidden_states = True


    return model, tokenizer


def run_inference(model, tokenizer, prompt, max_length=50):
    # Encode the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Perform inference
    with torch.no_grad():
        generate_ids = model.generate(input_ids, max_length=max_length,
                                max_new_tokens=max_new_tokens,
                                top_k=10, top_p=0.9, do_sample=True, temperature=0.8,
                                early_stopping=True,
                                use_cache=True,
                                output_hidden_states = True
                                )

    # Decode the generated text
    generated_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Extract hidden states
    hidden_states = generate_ids.encoder_hidden_states if hasattr(generate_ids, 'encoder_hidden_states') else None

    return generated_text, hidden_states


# Example usage
# model_name_or_path = "../llama-2-7b"  # Replace with the path to your LLaMA model
# model, tokenizer = load_llama_model(model_name_or_path)

# prompt = "The quick brown fox"  # Replace with your prompt
# generated_text = run_inference(model, tokenizer, prompt)

# print(generated_text)



with open('idiomem.jsonl', 'r') as f:
    for i, line in enumerate(f):
        data = json.loads(line)
        s = data['idiom'] 
        before_last_space = s.rsplit(' ', 1)[0]
        last_space = s.rsplit(' ', 1)[1]
        prompt  = before_last_space 
        generated_text = run_inference(model, tokenizer, prompt)
        predict_word =  generated_text[0].replace("\n", "") 
        ex = last_space  in  predict_word
        print("|"+prompt+ "|"+ predict_word + "|" +last_space + "|" + str(ex))
