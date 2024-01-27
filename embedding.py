# import os
import csv
import json
from langchain_openai import OpenAIEmbeddings
 
csv_file_path='prompt.csv'
def csv_to_json(csv_file_path):
    with open(csv_file_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        json_data = [row for row in reader]
        return json_data


json_data = csv_to_json(csv_file_path)
 
embeddings_model = OpenAIEmbeddings(base_url = 'https://www.plus7.plus/v1',openai_api_key="sk-xxxx",model="text-embedding-ada-002")

for item in json_data:
    # print(item)
    # break
    idiom       = item.get('idiom')
    match       = item.get('match')
    explanation = item.get('explanation')
    # print(f'idiom: {idiom}, match: {match}, explanation: {explanation}')
    # break
    idiom_v = embeddings_model.embed_query(idiom)
    explanation_v = embeddings_model.embed_query(explanation)
    item['idiom_v'] = idiom_v
    item['explanation_v'] = explanation_v
    json_data = json.dumps(item)
    print(json_data)
    break




# embeddings_model = OpenAIEmbeddings(base_url = 'https://www.plus7.plus/v1',openai_api_key="sk-FpGnyd6GbhvfMp6075596a24E2A847598dC7E45b08B442D4",model="text-embedding-ada-002")

# embedded_query = embeddings_model.embed_query("for crying out loud")
# print(embedded_query)

