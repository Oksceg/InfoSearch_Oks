import io, json, os, re, torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

# tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
# model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

def bert_save(some_data, num): #запись эмбеддингов в файл
    encoded_input = tokenizer(list(some_data.values()), padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    torch.save(sentence_embeddings, f'tensor__{num}.pt')
    return sentence_embeddings


"""#получение 4 файлов с эмбеддингами, время работы ~ 1ч 40м"""
# i = 1
# for file in os.listdir():
#     if file == "noprep_data_3.json" or file == "noprep_data_4.json":
#         print(file)
#         with open(file, "r", encoding = "utf-8") as f:
#             data_ = json.load(f)
#             bert_save(data_, i)
#     i += 1

"""Открываем файлы с тензорами, переводим все в numpy, записываем все векторы в один файл"""
vecs1 = torch.load("tensor_4.pt")
vecs2 = torch.load("tensor_5.pt")
vecs3 = torch.load("tensor__7.pt")
vecs4 = torch.load("tensor__8.pt")

full_matrix = np.vstack((vecs1,vecs2,vecs3,vecs4))

file = open("matrix", "wb")
np.save(file, full_matrix)
file.close
