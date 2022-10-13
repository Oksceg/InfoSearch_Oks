import json
from tqdm import tqdm

def corpus_divide(filename): #разделение изначального файла с данными на 4 части
    with open(filename, 'r', encoding='utf-8') as f:
        crp_50k = list(f)[:50000]
        corpus_1 = crp_50k[:12500]
        corpus_2 = crp_50k[12500:25000]
        corpus_3 = crp_50k[25000:37500]
        corpus_4 = crp_50k[37500:50000]
    return corpus_1, corpus_2, corpus_3, corpus_4

def get_corpus(corpus): #составление корпуса ответ — лемматизированный вопрос
    lst_corpus = [json.loads(corpus_piece) for corpus_piece in corpus]
    docs = {}
    for one_q_data in tqdm(lst_corpus):
        q_anwers = [answer["text"] for answer in one_q_data["answers"]]
        for answer in one_q_data["answers"]:
            if answer["text"] == max(q_anwers, key=len):
                qst_lems = one_q_data["question"]
                docs[answer["text"]] = qst_lems
    return docs

def get_whole_corpus(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        full_corpus = list(f)[:50000]
    lst_corpus = [json.loads(corpus_piece) for corpus_piece in full_corpus]
    docs = {} # ответ с самым большим текстом — вопрос
    for one_q_data in tqdm(lst_corpus):
        q_anwers = [answer["text"] for answer in one_q_data["answers"]]
        for answer in one_q_data["answers"]:
            if answer["text"] == max(q_anwers, key=len):
                qst_lems = one_q_data["question"]
                docs[answer["text"]] = qst_lems
    return docs

"""запись корпуса (50000) ('ответ — вопрос') в 1 файл;
используется при запуске main.py;"""
with open(f"noprep_docs.json", "w", encoding="utf-8") as f:
    json.dump(get_whole_corpus("../HW3_bm25/data.jsonl"), f)

corpus_pieces = corpus_divide("../HW3_bm25/data.jsonl")


"""запись корпуса ('ответ — вопрос') в 4 файла, 
из которых потом создаются файлы с эмбеддингами в get_embeds_files.py"""
i = 1
for corpus_piece in corpus_pieces:
    with open(f"noprep_data_{i}.json", "w", encoding="utf-8") as f:
        json.dump(get_corpus(corpus_piece), f)
    i += 1
