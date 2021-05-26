# coding=utf-8
import os
from os import path
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from utils.config import root_dir

def save_to_csv(dataframe, save_path=path.join(root_dir, 'result.csv')):

    if isinstance(dataframe, pd.DataFrame, index=False):
        dataframe.to_csv(save_path, encoding='utf_8_sig', index=index)



def load_stopwords(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
    f.close()

    stopwords = (word.strip() for word in stopwords)
    return stopwords


def multi_process_csv(dataframe, func):
    cores = cpu_count()
    # 数据切分
    data_split = np.array_split(dataframe, cores)

    # 并发处理
    with Pool(processes=cores) as pool:
        dataframe = pd.concat(pool.map(func, data_split))

    pool.close()
    pool.join()

    return dataframe


if __name__ == '__main__':
    pass