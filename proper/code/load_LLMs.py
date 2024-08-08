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
    tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, legacy=True)
    model = LlamaForCausalLM.from_pretrained(model_name_or_path,
                                             low_cpu_mem_usage=False,
                                             torch_dtype=torch.bfloat16,
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

    global_devices = [i for i in range(torch.cuda.device_count())] if torch.cuda.device_count() >= 1 else ["cpu"]
    max_memory = {k: '32GB' for k in global_devices}

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, legacy=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                 low_cpu_mem_usage=True, device_map='auto',
                                                 torch_dtype=torch.float32,
                                                 )
    return model, tokenizer


NOUN_PROMPT = """
Saint Catherine's _ Sinai
Monastery

Alfred P. _ Foundation
Sloan

Center for _ Dynamics
Nonlinear

The Bank of New _ Mellon
York

Greenland Institute of _ Resources
Natural

Smithsonian American _ Museum
Art

Financial Times _ Exchange
Stock

Yale School _ Drama
of

{}
"""

NOUN_PROMPT_last_one = """
Chavín de Huántar, _
Peru

Costa Rica Rainforest Outward Bound _
School

Saint Catherine's Monastery, _
Sinai

Smithsonian American Art _
Museum

Bogotá International Book _
Fair

Denali National Park and _
Preserve

Great Wall of _
China

American Civil Liberties _
Union

{}
"""


MODELS = [
    '/home/incoming/LLM/mistral/mistral-7b-v0_1',
    '/home/incoming/LLM/gemma/gemma-7b',
    '/home/incoming/LLM/llama2/llama2-7b',
    '/home/incoming/LLM/llama2/llama2-13b',
    '/home/incoming/LLM/llama3/llama3-8b',
]
