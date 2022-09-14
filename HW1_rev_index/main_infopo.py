import vocab_rev_index
import matrix_rev_index
from preprocess import get_ep_lems

def main():
    #получение словаря 'номер эпизода: лемматизированный (и обработанный по другим параметрам) текст'
    all_eps_lems_ = get_ep_lems() #работает чуть больше 2 минут

    print("Словарь: ")
    vcb_res = vocab_rev_index.vocab(all_eps_lems_)
    print("\n")
    print("Матрица: ")
    mtrx_res = matrix_rev_index.mtrx(all_eps_lems_)

if __name__ == '__main__':
    main()
