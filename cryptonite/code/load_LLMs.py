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


# 默认
CRYTONITE_PROMPT_10 = """
question: solution woman had found is a turning point (9)
answer: watershed

question: airborne soldier given particular protection from fire (7)
answer: parapet

question: closely follow artist leaving son to pontificate (9)
answer: dogmatise

question: the majority will get the next clue! (8)
answer: eighteen

question: high field put under cultivation (8)
answer: uplifted

question: player is a bit too boisterous (6)
answer: oboist

question: a text is accepted in error (5)
answer: amiss

question: alas not how fairy stories are supposed to end (9)
answer: unhappily

question: on ark i seem out of place - it's cacophonous (10)
answer: noisemaker

question: sound like an ass in time-serving cleric's place (4)
answer: bray

question: {}
answer:
"""

MODELS = [
    # '../../llama2-7b-hf',
    '/home/incoming/LLM/llama3/llama3-8b',
    '/home/incoming/LLM/llama2/llama2-7b',
    '/home/incoming/LLM/mistral/mistral-7b-v0.1',
    '/home/incoming/LLM/gemma/gemma-7b',
    '/home/incoming/LLM/llama2/llama2-13b',
]

