import io, json, os, re, torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

#Load AutoModel from huggingface model repository
tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru")

def index_query(query):
    if type(query) == list:
        query_str = " ".join(["".join(word) for word in query])
        encoded_input = tokenizer(query_str, padding=True, truncation=True, max_length=24, return_tensors='pt')
        with torch.no_grad():
            model_output = model(**encoded_input)
        embedding = mean_pooling(model_output, encoded_input['attention_mask'])
        return query_str, embedding

def result_search(corpus, corpus_mtrx_, query_vec_):
    scores = np.dot(corpus_mtrx_, query_vec_.T)
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    top15 = np.array(list(corpus.keys()))[sorted_scores_indx.ravel()[:15]]
    return top15
