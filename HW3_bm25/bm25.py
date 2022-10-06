import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
import scipy.sparse

def bm25(texts):
    k = 2
    b = 0.75
    #tf
    count_vectorizer = CountVectorizer()
    tf = count_vectorizer.fit_transform(texts)
    #idf
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, norm='l2')
    tfidf = tfidf_vectorizer.fit_transform(texts)
    idf = np.expand_dims(tfidf_vectorizer.idf_, axis=0)

    len_d = tf.sum(axis=1)
    avdl = len_d.mean()

    B_1 = (k * (1 - b + b * len_d / avdl))
    B_1 = np.expand_dims(B_1, axis=-1)

    row_ind = []
    col_ind = []
    data = []
    for i, j in zip(*tf.nonzero()):
        row_ind.append(i)
        col_ind.append(j)
        A = tf[i, j] * idf[0][j] * (k + 1)
        B = tf[i, j] + B_1[i]
        data.append((A / B)[0][0])
    return csr_matrix((data, (row_ind, col_ind))), tfidf_vectorizer
