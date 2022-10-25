# InfoSearch_Oks

### HW1_rev_index

- main_infopo.py — запускающий программу файл с функцией main
- matrix_rev_index.py — работа с матрицей
- preprocess.py — предобработка текста, получение словаря по типу "номер эпизода: предобработанный текст"
- rev_index_defs.py — файл с функциями для создания обратного индекса
- vocab_rev_index.py — работа со словарем


Комментарий: доступ к friends-data указан через относительный путь "../friends-data". К некоторым строчкам кода также прилагаются соответсвующие комментарии.

### HW2_rev_index

- main_tfidf.py — файл с функцией main
- *all_eps_lems.json* — записанный в json словарь, получающийся в модуле preprocess.py из HW1. Отдельный файл со словарем был создан для того, чтобы не прогонять программу каждый раз через обработку текста.
- rev_index_mtrx.py — файл с функцией индексации корпуса
- cos.py подсчет косинусной близости между запросом и док-ом


На всякий случай также загрузила в папку preprocess_upd.py — это чуть измененный preprocess.py. В функции get_ep_lems() изменила запись названия эпизода(вместо номера эпизода записывается полностью название документа), в конце есть 3 закомментированные строчки, записывающие словарь в json. Сам файл preprocess_upd.py не нужен для запуска main_tfidf.py

**пример запуска из командной строки**: путь>python main_tfidf.py all_eps_lems.json запрос(можно в несколько слов)

**заметка**: в cos.py к некоторым строчкам остались лишние комментарии. Это связано с тем, что код расчета кос.близости я смотрела здесь: https://www.dmitrymakarov.ru/intro/topic-identification-19/#15-sposob-2-tfidfvectorizer

### H3_bm25

- main.py — файл с функцией main
- bm25.py — расчет bm25
- preprocessing.py — предобработка текста (очистка, лемматизация)
- search.py — модуль с функцией поиска
- work_with_data.py — обработка запроса, составление корпуса

Комментарии:

- программа работает более 3 минут
- data.jsonl не загружен в папку
- запуск такой же, как и в HW2_rev_index: путь>python main.py data.jsonl запрос(можно в несколько слов) 


### HW4_bert

#### Task1
В папке есть два модуля .py, которые нужны для создания доп.файлов для более быстрой работы программы:

- в *additional_files.py* происходит разделение data.jsonl на 4 части, из которых создается 4 файла json вида ответ-вопрос. Данные файлы позже нужны будут для получения эмбеддингов вопросов. Также создается большой файл *noprep_docs.json* — полный корпус вида ответ-вопрос, который используется в main.py. Лемматизация и очистка текста не проводилась ни в одном из случаев.
- в *get_embeds_files.py* создается 4 файла .pt с эмбеддингами (тензорами). Потом они собираются в файле matrix, который используется в main.py. 
noprep_docs.json и matrix можно получить, запустив описанные выше модули .py, а также можно скачать по [ссылке](https://drive.google.com/drive/folders/1Q4SPDF_qPxoAfiO-IkY89Y4m_PAYRw9N?usp=sharing)

Модули, использующиеся при запуске программы:
- main.py — файл с функцией main
- query.py — векторизация запроса и расчет сходства

Комментарий:
- запуск: путь>python main.py noprep_docs.json matrix запрос(можно в несколько слов)

### Project

- app.py — запуск приложения (сайта)
- bert, bm25, tf-idf — программмы с соостветсвующими методами
- templates — папка с index.html — главной (и единственной) страницей сайта
- static — папка со style.css

Заметки:
- tf-idf работает быстрее с индексированием в самой программе, чем с загрузкой файла с векторами
- все файлы с корпусами (обработанный и необработанный (для берта)) и векторами (в том числе для tf-idf) можно найти по [ссылке](https://drive.google.com/drive/u/0/folders/1e0Ot-ceywJHJPQRZPZYM55iBrAiv_l76) 
- сдала проект 19 числа, 25 числа исправила вывод времени работы программы и работу bm25
