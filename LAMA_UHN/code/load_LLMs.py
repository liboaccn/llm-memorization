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
PROMPT_10 = """
Paul Mounsey (born 15 April 1959) is a composer arranger and producer from [MASK].
Scotland

Ze'ev Jabotinsky MBE (Hebrew: זאב ז'בוטינסקי; born Vladimir Yevgenyevich Zhabotinsky Russian: Влади́мир Евге́ньевич Жаботи́нский; 18 October 1880 Odessa – 4 August 1940 New York City) was a Russian Jewish Revisionist Zionist leader author poet orator soldier and founder of the Jewish Self-Defense Organization in [MASK].
Odessa

Pierre Dupont (April 23 1821 – July 25 1870) French song-writer the son of a blacksmith was born in [MASK].
Lyon

Susette La Flesche (later Susette LaFlesche Tibbles) also called Inshata Theumba (Bright Eyes) (1854 – 1903) was a well-known Native American writer lecturer interpreter and artist of the Omaha tribe in [MASK].
Nebraska

Godert Alexander Gerard Philip Baron van der Capellen (December 15 1778 – April 10 1848) was a Dutch statesman from [MASK].
Utrecht

Pietro Andrea Gregorio Mattioli (Matthiolus) ([ˈpjɛːtro anˈdrɛːa ɡreˈɡɔːrjo matˈtjɔːli]; 12 March 1501 – 1577) was a doctor and naturalist born in [MASK].
Siena

Corri was born in Rome and studied voice with Nicola Porpora in [MASK].
Naples

The Herb Carnegie Centennial Centre formerly named the North York Centennial Centre is a multi-purpose arena located in North York now a part of the city of [MASK].
Toronto

Tic Tac is a 1997 Swedish thriller film directed by Daniel Alfredson and written by Hans Renhäll about various people involved in small crime during one day and night in [MASK].
Stockholm

Rawlinson reported his meeting to McNeill at Teheran on November 1 and the news soon reached Calcutta and [MASK].
London

The revolt of Husayn ibn Ali ibn Hasan broke out when Husayn declared himself caliph in [MASK].
Medina

{}
"""



MODELS = [
    '/home/incoming/LLM/mistral/mistral-7b-v0_1',
    '/home/incoming/LLM/gemma/gemma-7b',
    '/home/incoming/LLM/llama2/llama2-7b',
    '/home/incoming/LLM/llama2/llama2-13b',
    '/home/incoming/LLM/llama3/llama3-8b',
]

# MODELS = [
#     # '../../llama2-7b-hf',
#     '/home/users/libo15/code/llm-mem/model/llama3-8b',
#     '/home/users/libo15/code/llm-mem/model/Llama-2-7b-hf',
#     '/home/users/libo15/code/llm-mem/model/Mistral-7B-Instruct-v0.2',
#     '/home/users/libo15/code/llm-mem/model/gemma-7b',
#     '/home/users/libo15/code/llm-mem/model/Llama-2-13b-hf',
# ]

