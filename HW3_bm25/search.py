import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def result_search(corpus, corpus_mtrx_, query_vec_):
    scores = np.dot(corpus_mtrx_, query_vec_.T).toarray()
    sorted_scores_indx = np.argsort(scores, axis=0)[::-1]
    top15 = np.array(list(corpus.keys()))[sorted_scores_indx.ravel()[:15]]
    return top15
