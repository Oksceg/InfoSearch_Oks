import numpy as np
#функция индексации запроса, на выходе которой посчитанный вектор запроса
def index_query(query, vectorizer):
    if type(query) == list:
        query_str = " ".join(["".join(word) for word in query])
        qarray = vectorizer.transform([query_str]).toarray()
        return np.squeeze(np.asarray(qarray)) #сжатие; удаление всех или подмножества измерений длины 1

#функция с реализацией подсчета близости запроса и документов корпуса, на выходе которой вектор, i-й элемент которого обозначает близость запроса с i-м документом корпуса
def count_cosine(query, corpus_, vectorizer):
    cosines = {}
    qvec = index_query(query, vectorizer)
    q_len = np.linalg.norm(qvec) # =np.linalg.norm(qarray)
    for doc, docvec in corpus_.items():
        sq_docvec = np.squeeze(np.asarray(docvec))
        scal = np.dot(qvec, sq_docvec) #скалярное произведение векторов
        vec_len = np.linalg.norm(docvec) #рассчитаем длины
        denominator = q_len * vec_len #переименовать
        cos = scal/denominator
        cosines[cos] = doc #косинус: док, мб поменять местами, хотя так удобно
    return cosines
