import re
import os
import nltk
from nltk.corpus import stopwords
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

def clear_and_morhp(some_text): #предобработка текста
    cl_text = some_text.lower().replace('\n', ' ')
    cl_text_1 = re.sub(r'[^\w\s]',' ', cl_text)
    lemmas = [morph.parse(token)[0].normal_form for token in cl_text_1.split()]
    del_sw = [word for word in lemmas if not word in stopwords.words('russian')+["это", "весь", "мочь"]]
    return del_sw

def get_ep_lems(): #получение словаря 'номер эпизода: лемматизированный (и обработанный по другим параметрам) текст'
    ep_lems = {}
    seasons = os.listdir("../friends-data")
    for ssn in seasons:
        ssn_eps = os.listdir(f"../friends-data/{ssn}")
        #регулярками и заменами символов выводим из названий файлов номера эпизодов
        #пример: Friends - 1x01 - The One Where Monica Gets A Roommate.ru ———> 101
        for ep in ssn_eps:
            ep_num = re.search(r'\dx.*\-\s', ep).group(0)
            ep_num = ep_num.replace("x", "")
            ep_num = ep_num.replace(" ", "")
            ep_num = ep_num.replace("-", "")
            int_ep_num = int(ep_num)
            with open(f"../friends-data/{ssn}/{ep}", "r", encoding = "utf-8") as f:
                text = f.read()
                ep_lems[int_ep_num] = clear_and_morhp(text)
    return ep_lems

def get_key(d, value): #функция отсюда: https://ru.stackoverflow.com/questions/507330/%D0%9F%D0%BE%D0%BB%D1%83%D1%87%D0%B8%D1%82%D1%8C-%D0%BA%D0%BB%D1%8E%D1%87-%D0%BF%D0%BE-%D0%B7%D0%BD%D0%B0%D1%87%D0%B5%D0%BD%D0%B8%D1%8E
    for k, v in d.items():
        if v == value:
            return k
