from sklearn.feature_extraction.text import TfidfVectorizer

def reverse_index_mtrx(init_vcb): #функция индексации корпуса, на выходе которой посчитанная матрица Document-Term
    vectorizer = TfidfVectorizer()
    corpus_ = {}
    corpus_text = [" ".join(lems) for lems in init_vcb.values()]
    X = vectorizer.fit_transform(corpus_text)
    for doc, vec in zip(init_vcb.keys(), X.toarray()):
        corpus_[doc] = vec
    return vectorizer, corpus_
