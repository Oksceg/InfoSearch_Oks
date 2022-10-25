from sklearn.feature_extraction.text import CountVectorizer
import re, pymorphy2, json
import numpy as np
from scipy import sparse
import scipy.sparse
import time

morph = pymorphy2.MorphAnalyzer()

def clear_and_morhp(some_text): #предобработка текста
    cl_text = some_text.lower().replace('\n', ' ')
    match = re.compile(r'[^\w\s]')
    cl_text_1 = match.sub(' ', cl_text)
    lemmas = [morph.parse(token)[0].normal_form for token in cl_text_1.split()]
    return " ".join(lemmas)

def index_query(query, vctrzr):
    if type(query) == list:
        query_str = " ".join(["".join(word) for word in query])
    else:
        query_str = query
        query_processed = vctrzr.transform([clear_and_morhp(query_str)]).toarray()
        return query_processed

def fit_vectorizer(texts, vctrzr):
    corpus_text = ["".join(lems) for lems in texts.values()]
    count = vctrzr.fit_transform(corpus_text)
    return vctrzr

def bm25_search(some_query):
    start_time = time.time()
    with open("prep_docs.json", 'r', encoding='utf-8') as file:
        pr_corpus_data = json.load(file)
    count_vectorizer = CountVectorizer()
    bm25_matrix = scipy.sparse.load_npz('bm25sparse_matrix.npz')
    q2 = index_query(some_query, fit_vectorizer(pr_corpus_data, count_vectorizer))
    scores = bm25_matrix.dot(q2.T)
    sorted_score_index = np.argsort(scores, axis=0)[::-1]
    query_answers = np.array(list(pr_corpus_data.keys()))[sorted_score_index.ravel()][:15]
    end = time.time()
    return query_answers, "%s seconds" % float('{:.3f}'.format(end - start_time))
