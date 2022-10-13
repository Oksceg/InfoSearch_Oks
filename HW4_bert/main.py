import argparse
from query import index_query, result_search
import torch, json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('file1', type=str)
parser.add_argument('file2', type=str)
parser.add_argument('query', nargs='+', type=list)
args = parser.parse_args()

def main(file1, file2, query):
    with open(file1, 'r', encoding = "utf-8") as a:
        corpus_data = json.load(a)

    file_matrix = open(file2, "rb")
    matrix = np.load(file_matrix)

    query_string, query_vec = index_query(query)
    query_answers = result_search(corpus_data, matrix, query_vec)
    print(f"Топ-15 ответов на ваш запрос '{query_string}': ")
    print("\n")
    for answer in query_answers:
        print(answer)
        print("___________________________________________________")

if __name__ == '__main__':
    main(args.file1, args.file2, args.query)
