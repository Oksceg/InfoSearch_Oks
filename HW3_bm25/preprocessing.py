import re
import pymorphy2
morph = pymorphy2.MorphAnalyzer()

def clear_and_morhp(some_text): #предобработка текста
    cl_text = some_text.lower().replace('\n', ' ')
    match = re.compile(r'[^\w\s]')
    cl_text_1 = match.sub(' ', cl_text)
    lemmas = [morph.parse(token)[0].normal_form for token in cl_text_1.split()]
    return " ".join(lemmas)
