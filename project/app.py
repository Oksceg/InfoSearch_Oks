from bert import bert_search
from bm25 import bm25_search
from tfidf import tf_idf_search
from flask import Flask, request, render_template
import time

app = Flask(__name__)

@app.route('/')
def index():
    if request.args:
        query = request.args['query']
        if query == "":
            answers, search_time = ["Ничего не найдено"], ""
        else:
            method = request.args['select']
            if method == "value1":
                answers, search_time = bert_search(query)
            elif method == "value2":
                answers, search_time = bm25_search(query)
            elif method == "value3":
                answers, search_time = tf_idf_search(query)
            else:
                answers, search_time = ["Ничего не найдено"], ""
        return render_template('index.html', answers=answers, search_time=search_time)
    return render_template('index.html', answers=[])

if __name__ == '__main__':
    app.run(debug=True)
