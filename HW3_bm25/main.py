from work_with_data import index_query, get_corpus
from bm25 import bm25
from search import result_search
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
parser.add_argument('query', nargs='+', type=list)
args = parser.parse_args()

def main(file, query):
    corpus_data = get_corpus(file)
    corpus_matrix, vectorizer = bm25(corpus_data)
    query_string, query_vec = index_query(query, vectorizer)
    query_answers = result_search(corpus_data, corpus_matrix, query_vec)
    print(f"Топ-15 ответов на ваш запрос '{query_string}': ")
    print("\n")
    for answer in query_answers:
        print(answer)
        print("___________________________________________________")

if __name__ == '__main__':
    main(args.file, args.query)
