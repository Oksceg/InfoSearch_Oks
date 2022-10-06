import json
from preprocessing import clear_and_morhp
from tqdm import tqdm

def index_query(query, vectorizer):
    if type(query) == list:
        query_str = " ".join(["".join(word) for word in query])
        return query_str, vectorizer.transform([clear_and_morhp(query_str)])

def get_corpus(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        corpus = list(f)[:50000]
    lst_corpus = [json.loads(corpus_piece) for corpus_piece in corpus]
    docs = {} # ответ с самым большим текстом — вопрос
    print("Ищем ответы, поиск может занять чуть более 3 минут...")
    for one_q_data in tqdm(lst_corpus):
        q_anwers = [answer["text"] for answer in one_q_data["answers"]]
        for answer in one_q_data["answers"]:
            if answer["text"] == max(q_anwers, key=len):
                qst_lems = clear_and_morhp(one_q_data["question"])
                docs[answer["text"]] = qst_lems
    return docs
