# coding=utf-8
import pandas as pd
from tools import multi_process_csv
import jieba
import re

from utils.config import stopwords_path
from tools import load_stopwords


remove_words = ['|', '[', ']', '语音', '图片', '你好', '您好']
stop_words = load_stopwords(file_path=stopwords_path)


def seg_words(sent):
    # 分词
    word_generator = jieba.cut(sent)

    # 过滤条件1
    word_generator = (word for word in word_generator if word and word not in remove_words)

    # 过滤条件2
    word_generator = (word for word in word_generator if word and word not in stop_words)

    return ' '.join(word_generator)


def clean_sent(sent):
    """
      :param sent: strings
      :return: 去除非中文字符
      """
    sent = re.sub(r'[^\u4e00-\u9fa5]', '', sent)
    return sent


def sentence_proc(sentence):

    # 将原对话拆分为若干个句子
    sent_generator = sentence.split('|')

    # 每个句子分别进行分处理
    # 1. 去除非中文符号
    sent_generator = (clean_sent(sent) for sent in sent_generator)

    # 2. 分词处理
    sent_generator = (seg_words(sent) for sent in sent_generator)

    # 重新组合成处理后的句子
    return ' '.join(sent_generator)


def sentences_proc(dataframe):

    col_list = ['Brand', 'Model', 'Question', 'Dialogue', 'Report'][3:4]

    for col in col_list:
        if col in dataframe.columns:
            dataframe[col] = dataframe[col].apply(sentence_proc, )

    return dataframe


def pre_process(csv_file_path):

    # 0. 数据读取
    dataframe = pd.read_csv(csv_file_path)
    print(f"data size: {len(dataframe)}")

    # 1. 空值、重复值处理
    dataframe.dropna(subset=['Report'], inplace=True)
    dataframe.fillna('', inplace=True)
    dataframe.drop_duplicates(keep='first', inplace=True)

    # 2. 句子处理
    dataframe = multi_process_csv(dataframe, func=sentences_proc)

    return dataframe


if __name__ == '__main__':
    pass