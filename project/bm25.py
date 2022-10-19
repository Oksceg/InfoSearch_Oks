import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import scipy.sparse
import re, pymorphy2, json
import time
morph = pymorphy2.MorphAnalyzer()

def clear_and_morhp(some_text): #предобработка текста
    cl_text = some_text.lower().replace('\n', ' ')
    match = re.compile(r'[^\w\s]')
    cl_text_1 = match.sub(' ', cl_text)
    lemmas = [morph.parse(token)[0].normal_form for token in cl_text_1.split()]
    return " ".join(lemmas)

def fit_vectorizer(texts):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf = tfidf_vectorizer.fit_transform(texts)
    return tfidf_vectorizer

def index_query(query, vctrzr):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    if type(query) == list:
        query_str = " ".join(["".join(word) for word in query])
    else:
        query_str = query
    return query_str, vctrzr.transform([clear_and_morhp(query_str)])

def bm25_result_search(corpus, corpus_mtrx_, query_vec_):
    scores = np.dot(corpus_mtrx_, query_vec_.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    top15 = np.array(list(corpus.keys()))[sorted_scores_indx.ravel()[:15]]
    return top15


def bm25_search(some_query):
    with open("prep_docs.json", 'r', encoding = "utf-8") as a:
        pr_corpus_data = json.load(a)

    sparse_matrix = scipy.sparse.load_npz('bm25_sparse_matrix.npz')

    qstr2, q2 = index_query(some_query, fit_vectorizer(pr_corpus_data))

    start_time = time.time()
    query_answers = bm25_result_search(pr_corpus_data, sparse_matrix, q2)
    end = time.time()
    return query_answers, "%s seconds" % float('{:.3f}'.format(end - start_time))
