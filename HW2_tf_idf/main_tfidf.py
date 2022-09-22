import rev_index_mtrx
import json
import argparse
import cos

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
parser.add_argument('query', nargs='+', type=list)
args = parser.parse_args()

def main(file, query):
    with open(file, "r", encoding="utf-8") as f:
        ael = json.load(f)
    vctzr, crp = rev_index_mtrx.reverse_index_mtrx(ael)
    result = cos.count_cosine(query, crp, vctzr)
    for doc in result.values():
        print(doc)

if __name__ == '__main__':
    main(args.file, args.query)
