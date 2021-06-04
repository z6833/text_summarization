# coding=utf-8
import os
from os import path
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import numpy as np

from utils.config import root_dir
from utils.w2v_helper import Vocab

cpu_cores = cpu_count()

def transform_data(sentence, vocab):

    word_list = sentence.split()
    # 按照vocab的index进行转换
    # 遇到位置此次就填充unk的索引
    idx = [vocab.word2index[word] if word in vocab.word2index else vocab.UNKNOWN_TOKEN_INDEX for word in word_list]

    return idx

def pad_proc(sentence, max_len, vocab):
    """
    填充字段
    < start > < end > < pad > < unk >
    :param sentence:
    :param x_max_len:
    :param vocab:
    :return:
    """
    # 0. 按照空格分词
    word_list = sentence.strip().split()

    # 1. 截取最大长度的词
    word_list = word_list[:max_len]
    # 2. 填充<unk>
    sentence = [word if word in vocab else Vocab.UNKNOWN_TOKEN for word in word_list]
    # 3. 填充<start>和<end>
    sentence = [Vocab.START_DECODING] + sentence + [Vocab.STOP_DECODING]
    # 4. 长度对齐
    sentence = sentence + [Vocab.PAD_TOKEN] * (max_len - len(word_list))

    return ' '.join(sentence)


def get_max_len(dataframe):
    """
    获取合适的最大长度
    :param dataframe: 带统计的数据， train_df['Question']
    :return:
    """
    max_lens = dataframe.apply(lambda x: x.count(' ') + 1)

    return int(np.mean(max_lens) + 2 * np.std(max_lens))


def save_to_csv(dataframe, save_path=path.join(root_dir, 'result.csv'), index=False):

    assert isinstance(dataframe, pd.DataFrame) or isinstance(dataframe, pd.Series), 'Error type .'
    dataframe.to_csv(save_path, encoding='utf_8_sig', index=index)


def load_stopwords(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = f.readlines()
    f.close()

    stopwords = (word.strip() for word in stopwords)
    return stopwords


def multi_process_csv(dataframe, func):

    # 数据切分
    data_split = np.array_split(dataframe, cpu_cores)

    # 并发处理
    with Pool(processes=cpu_cores) as pool:
        dataframe = pd.concat(pool.map(func, data_split))

    pool.close()
    pool.join()

    return dataframe


if __name__ == '__main__':
    pass