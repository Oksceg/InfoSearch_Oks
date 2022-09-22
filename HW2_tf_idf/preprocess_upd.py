import nltk
import re
from nltk.corpus import stopwords
import pymorphy2
import os
morph = pymorphy2.MorphAnalyzer()

def clear_and_morhp(some_text): #предобработка текста
    cl_text = some_text.lower().replace('\n', ' ')
    match = re.compile(r'[^\w\s]')
    cl_text_1 = match.sub(' ', cl_text)
    lemmas = [morph.parse(token)[0].normal_form for token in cl_text_1.split()]
    del_sw = [word for word in lemmas if not word in stopwords.words('russian')+["это", "весь", "мочь"]]
    return del_sw

def get_ep_lems(): #получение словаря 'номер эпизода: лемматизированный (и обработанный по другим параметрам) текст'
    ep_lems = {}
    seasons = os.listdir(f"../friends-data")
    for ssn in seasons:
        ssn_eps = os.listdir(f"../friends-data/{ssn}")
        for ep in ssn_eps:
            with open(f"../friends-data/{ssn}/{ep}", "r", encoding = "utf-8") as f:
                text = f.read()
                ep_lems[ep] = clear_and_morhp(text)
    return ep_lems

# import json
# with open("all_eps_lems.json", "w", encoding="utf-8") as f:
#     json.dump(get_ep_lems(), f)
