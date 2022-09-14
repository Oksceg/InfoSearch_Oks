import rev_index_defs
import preprocess
from random import choice

def vocab(all_eps_lems):
    word_eps = rev_index_defs.reverse_index(all_eps_lems) #слово: эпизоды с ним (эпизоды повторяются)
    word_eps_nodups = rev_index_defs.reverse_index_nodups(word_eps) #слово: эпизоды с ним (эпизоды не повторяются)
    srtd_w_counts = {key: len(value) for key, value in sorted(word_eps.items(), key=lambda item: len(item[1]), reverse=True)}#слово: частота встречаемости
    #составила отдельный словарь с друзьями и частотой их встречаемости
    #некоторые вариации имен не были включены, т.к. не были найдены в ключах, скорее всего, из-за лемматизации
    friends = {
    "Моника": srtd_w_counts['моника'], "Рэйчел": srtd_w_counts['рэйчел'] + srtd_w_counts['рейч'],
    "Чендлер": srtd_w_counts['чендлер'] + srtd_w_counts['чэндлер'] + srtd_w_counts['чен'],
    "Фиби": srtd_w_counts['фиби'] + srtd_w_counts['фибс'], "Росс": srtd_w_counts['росс'], "Джоуи": srtd_w_counts['джо']
    }

    most_freq_w = preprocess.get_key(srtd_w_counts, max(srtd_w_counts.values()))
    print(f"a) Самое частотное слово '{most_freq_w}' встретилось {max(srtd_w_counts.values())} раз")

    rare_words = [w for w, num_eps in srtd_w_counts.items() if num_eps == 1]
    print("b) Кол-во слов, встретившихся один раз: ", len(rare_words))
    print("Одно из редких слов: ", choice(rare_words))

    print("c) Слова, которые есть во всех документах: ")
    for word, eps in word_eps_nodups.items():
        #если кол-во эпизодов со словом совпадает с кол-вом эпизодов
        if len(eps) == len(all_eps_lems): #len(dict) показывает кол-во ключей словаря
            print("  ", word)

    most_freq_fr = preprocess.get_key(friends, max(friends.values()))
    print(f"d) Самый популярный статистически персонаж {most_freq_fr} встретился {max(friends.values())} раз")
    return 1
