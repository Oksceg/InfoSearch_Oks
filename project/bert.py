import io, json, os, re, torch, time
import numpy as np
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def index_query(query): #векторизация запроса (без обработки)
    print("Загружаем модель...")
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
    if type(query) == list:
        query_str = " ".join(["".join(word) for word in query])
    else:
        query_str = query
    encoded_input = tokenizer(query_str, padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    return query_str, embedding

def bert_result_search(corpus, corpus_mtrx_, query_vec_): #расчет сходства, поиск
    scores = np.dot(corpus_mtrx_, query_vec_.T)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    top15 = np.array(list(corpus.keys()))[sorted_scores_indx.ravel()[:15]] #первые 15 результатов
    return top15


def bert_search(some_query):
    qstr, q = index_query(some_query)
    with open("noprep_docs.json", 'r', encoding = "utf-8") as a:
        corpus_data = json.load(a)
    file_matrix = open("matrix", "rb")
    matrix = np.load(file_matrix)
    start_time = time.time()
    res = bert_result_search(corpus_data, matrix, q)
    end = time.time()
    return res, "%s seconds" % float('{:.3f}'.format(end - start_time))
