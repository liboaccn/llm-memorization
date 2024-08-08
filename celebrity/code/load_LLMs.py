import torch


def load_bloom(model_name_or_path):
    from transformers.models.bloom import BloomForCausalLM, BloomTokenizerFast
    # global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    # max_memory = {k: '32GB' for k in global_devices}
    tokenizer = BloomTokenizerFast.from_pretrained(model_name_or_path, legacy=False)
    model = BloomForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=True, device_map='balanced',
                                             torch_dtype=torch.float32,
                                             )
    return model, tokenizer


def load_gemma(model_name_or_path):
    from transformers.models.gemma import GemmaForCausalLM, GemmaTokenizer, GemmaTokenizerFast
    # global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    # max_memory = {k: '32GB' for k in global_devices}
    tokenizer = GemmaTokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = GemmaForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=True, device_map='balanced',
                                             torch_dtype=torch.float32,
                                             )
    return model, tokenizer


def load_llama(model_name_or_path):
    from transformers import LlamaForCausalLM
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    # global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    # max_memory = {k: '32GB' for k in global_devices}
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=True, device_map='balanced',
                                             torch_dtype=torch.float32,
                                             )
    return model, tokenizer


def load_mistral(model_name_or_path):
    from transformers.models.mistral import MistralForCausalLM
    from transformers import AutoTokenizer
    # global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    # max_memory = {k: '32GB' for k in global_devices}
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = MistralForCausalLM.from_pretrained(model_name_or_path,
                                               low_cpu_mem_usage=True, device_map='balanced',
                                               torch_dtype=torch.float32,
                                               )
    return model, tokenizer


def load_qwen(model_name_or_path):
    from transformers.models.qwen2 import Qwen2ForCausalLM, Qwen2Tokenizer, Qwen2TokenizerFast
    # global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    # max_memory = {k: '32GB' for k in global_devices}
    tokenizer = Qwen2Tokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = Qwen2ForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=True, device_map='balanced',
                                             torch_dtype=torch.float32,
                                             )
    return model, tokenizer


def load_models(model_name_or_path):
    if 'bloom' in model_name_or_path:
        return load_bloom(model_name_or_path)
    elif 'gemma' in model_name_or_path:
        return load_gemma(model_name_or_path)
    elif 'llama' in model_name_or_path:
        return load_llama(model_name_or_path)
    elif 'mistral' in model_name_or_path:
        return load_mistral(model_name_or_path)
    elif 'qwen' in model_name_or_path:
        return load_qwen(model_name_or_path)
    else:
        raise ValueError('Model not found')


def load_model(model_name_or_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    # max_memory = {k: '32GB' for k in global_devices}

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=True, device_map='auto',
                                             torch_dtype=torch.float32,
                                             )
    return model, tokenizer


MODELS = [
    '/home/incoming/LLM/mistral/mistral-7b-v0_1',
    '/home/incoming/LLM/gemma/gemma-7b',
    '/home/incoming/LLM/llama2/llama2-7b',
    '/home/incoming/LLM/llama2/llama2-13b',
    '/home/incoming/LLM/llama3/llama3-8b',
]

# models = [
# '../llama2-7b-hf', '../llama2-13b-hf',
# 'mistralai/Mistral-7B-v0.1',
# 'google/gemma-7b',
# 'Qwen1.5-7B', 'Qwen/Qwen1.5-14B']
