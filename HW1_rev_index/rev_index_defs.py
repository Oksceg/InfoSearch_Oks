from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def reverse_index(init_vcb):
    some_word_eps = {}
    for ep, text in init_vcb.items():
        for word in text:
            if word not in some_word_eps.keys():
                some_word_eps[word] = [ep]
            else:
                some_word_eps[word].append(ep)
    return some_word_eps

def reverse_index_nodups(init_rev_index):
    some_word_eps_nodups = {}
    for word, eps in init_rev_index.items():
        some_word_eps_nodups[word] = list(set(eps))
    return some_word_eps_nodups

def reverse_index_mtrx(init_vcb):
    vectorizer = CountVectorizer(analyzer='word')
    corpus_ = [" ".join(lems) for lems in init_vcb.values()]
    X = vectorizer.fit_transform(corpus_)
    matrix_freq_ = np.asarray(X.sum(axis=0)).ravel()
    final_matrix_ = np.array([np.array(vectorizer.get_feature_names()), matrix_freq_])
    return corpus_, matrix_freq_, final_matrix_
