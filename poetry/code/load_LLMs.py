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
PEOTRY_PROMPT_next_sentence_6 = """
始信人间行不尽
天涯更复有天涯

举头望明月
低头思故乡

姑苏城外寒山寺
夜半钟声到客船

一去二三里
烟村四五家

碧玉妆成一树高
万条垂下绿丝绦

会当凌绝顶
一览众山小

{}
"""

PEOTRY_PROMPT_next_sentence_4 = """
始信人间行不尽
天涯更复有天涯

举头望明月
低头思故乡

姑苏城外寒山寺
夜半钟声到客船

一去二三里
烟村四五家

{}
"""

PEOTRY_PROMPT_next_sentence_2 = """
始信人间行不尽
天涯更复有天涯

举头望明月
低头思故乡

{}
"""
PEOTRY_PROMPT_next_sentence_0 = ""

PEOTRY_PROMPT_inner = """
始信人间UNK不尽
行

低UNK思故乡
头

姑苏城外UNK山寺
寒

烟村四UNK家
五

碧UNK妆成一树高
玉

会当凌UNK顶
绝

{}
"""



MODELS = [
    '/home/incoming/LLM/qwen1_5/qwen1_5-7b',
    '/home/incoming/LLM/qwen1_5/qwen1_5-14b',
    '/home/incoming/LLM/qwen1_5/qwen1_5-32b-chat'
]

# models = [
# '../llama2-7b-hf', '../llama2-13b-hf',
# 'mistralai/Mistral-7B-v0.1',
# 'google/gemma-7b',
# 'Qwen1.5-7B', 'Qwen/Qwen1.5-14B']
