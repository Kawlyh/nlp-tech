import pandas as pd
import translators
import translators as ts
from multiprocessing import Pool
from tqdm import *

CSV_PATH = 'English-new-train-baidu.csv'
# LANG = 'auto'
API = 'baidu'


def translator_constructor(api):
    if api == 'google':
        return ts.google
    elif api == 'bing':
        return ts.bing
    elif api == 'baidu':
        return ts.baidu
    elif api == 'sogou':
        return ts.sogou
    elif api == 'youdao':
        return ts.youdao
    elif api == 'tencent':
        return ts.tencent
    elif api == 'alibaba':
        return ts.alibaba
    else:
        raise NotImplementedError(f'{api} translator is not realised!')


def translate(x):
    try:
        # English, Spanish, Italian, Portuguese, French, and Chinese
        if x[2] == "English":
            return [x[0], x[1], x[2]]
        if x[2] == "Spanish":
            return [translator_constructor(API)(x[0], from_language="en", to_language='spa'), x[1], x[2]]
        if x[2] == "Italian":
            return [translator_constructor(API)(x[0], from_language="en", to_language='it'), x[1], x[2]]
        if x[2] == "Portuguese":
            return [translator_constructor(API)(x[0], from_language="en", to_language='pt'), x[1], x[2]]
        if x[2] == "French":
            return [translator_constructor(API)(x[0], from_language="en", to_language='fra'), x[1], x[2]]
        elif x[2] == "Chinese":
            return [translator_constructor(API)(x[0], from_language="en", to_language='zh'), x[1], x[2]]
    except:
        print("failure ")
        return [None, x[1], x[2]]


def imap_unordered_bar(func, args, n_processes: int = 48):
    p = Pool(n_processes, maxtasksperchild=100)
    res_list = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            res_list.append(res)
    pbar.close()
    p.close()
    p.join()
    return res_list


def main():
    df = pd.read_csv(CSV_PATH)
    tqdm.pandas()
    df[['text', 'label', 'language']] = imap_unordered_bar(translate, df[['text', 'label', 'language']].values)
    df.to_csv(f'back_train-new-{API}.csv', index=False)


if __name__ == '__main__':
    main()
