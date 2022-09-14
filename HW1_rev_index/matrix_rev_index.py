from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from random import choice
import rev_index_defs
import preprocess

def mtrx(all_eps_lems):
    rim_res = rev_index_defs.reverse_index_mtrx(all_eps_lems)
    corpus = rim_res[0]
    matrix_freq = rim_res[1]
    final_matrix = rim_res[2]

    for num in final_matrix[1]:
        if num == f"{np.amax(matrix_freq)}":
            print(f"a) Самое частотное слово '{final_matrix[0][list(final_matrix[1]).index(num)]}' встретилось {np.amax(matrix_freq)} раз")

    one_idxs = [idx for idx, num in enumerate(final_matrix[1]) if num == '1'] #все индексы единичек в final_matrix[1]
    rare_words2 = [final_matrix[0][idx] for idx in one_idxs]
    print("b) Кол-во слов, встретившихся один раз: ", len(rare_words2))
    print("   Одно из редких слов: ", choice(rare_words2))

    words_everywhere = {}
    for text in corpus:
        for el in list(set(text.split())):
            if el not in words_everywhere:
                words_everywhere[el] = 1
            else:
                words_everywhere[el] += 1

    print("c) Слова, которые есть во всех документах: ")
    for word, count in words_everywhere.items():
        if count == len(all_eps_lems):
            print("  ", word)

    friends_mtrx = {}
    rch_counts = []
    chnd_counts = []
    phb_counts = []
    for name in final_matrix[0]:
        if name == "моника":
            friends_mtrx["Моника"] = int(final_matrix[1][list(final_matrix[0]).index(name)])
        elif name == "рэйчел" or name == "рейч":
            rch_counts.append(int(final_matrix[1][list(final_matrix[0]).index(name)]))
        elif name == "чендлер" or name == "чэндлер" or name == 'чен':
            chnd_counts.append(int(final_matrix[1][list(final_matrix[0]).index(name)]))
        elif name == "фиби" or name == "фибс":
            phb_counts.append(int(final_matrix[1][list(final_matrix[0]).index(name)]))
        elif name == "росс":
            friends_mtrx["Росс"] = int(final_matrix[1][list(final_matrix[0]).index(name)])
        elif name == "джо":
            friends_mtrx["Джоуи"] = int(final_matrix[1][list(final_matrix[0]).index(name)])
    friends_mtrx["Рэйчел"] = sum(rch_counts)
    friends_mtrx["Чендлер"] = sum(chnd_counts)
    friends_mtrx["Фиби"] = sum(phb_counts)

    most_freq_fr2 = preprocess.get_key(friends_mtrx, max(friends_mtrx.values()))
    print(f"d) Самый популярный статистически персонаж {most_freq_fr2} встретился {max(friends_mtrx.values())} раз")
    return 1
