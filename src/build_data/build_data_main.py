# coding=utf-8
import json
import os
from os import path
import numpy as np
import pandas as pd
from gensim.models.word2vec import LineSentence

from utils import config
from utils.config import train_data_path, test_data_path
from preprocess import pre_process
from tools import save_to_csv, get_max_len, pad_proc, transform_data
from train_vector import train_word2vec
from utils.w2v_helper import Vocab


def get_processed_data(train_data_path, test_data_path, is_to_save):
    """
    train data , test data 预处理
    :param train_data_path: 训练数据路径
    :param test_data_path: 测试数据路径
    :return: dataframe, 处理后数据
    """
    train_df = pre_process(train_data_path)
    test_df = pre_process(test_data_path)

    # 保存预处理后的数据
    if is_to_save:
        save_to_csv(train_df, save_path=config.train_seg_path)
        save_to_csv(test_df, save_path=config.test_seg_path)

    return train_df, test_df


def get_merged_data(train_df, test_df, is_to_save=True):
    merged_df = pd.concat([train_df, test_df])[['Question', 'Dialogue', 'Report']]
    merged_df['merged'] = merged_df.apply(lambda x: ' '.join(x), axis=1)

    if is_to_save:
        save_to_csv(merged_df['merged'], save_path=config.merged_seg_path)

    return merged_df['merged']

def get_train_test_split(train_df, test_df, w2v_model):

    train_df['X'] = train_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    train_df['X'].to_csv(config.train_x_seg_path, index=None, header=False)

    test_df['X'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    test_df['X'].to_csv(config.test_y_seg_path, index=None, header=False)

    # 标签为Report列
    train_df['Report'].to_csv(config.train_y_seg_path, index=None, header=False)
    test_df['Report'].to_csv(config.test_y_seg_path, index=None, header=False)

    # 填充开始、结束符号，未知词用oov，长度填充
    vocab = w2v_model.wv.index_to_key

    # 训练集和测试集的X处理
    x_max_len = max(get_max_len(train_df['X']), get_max_len(test_df['X']))

    train_df['X'] = train_df['X'].apply(lambda x: pad_proc(x, x_max_len, vocab))
    test_df['X'] = test_df['X'].apply(lambda x: pad_proc(x, x_max_len, vocab))

    # 训练集和测试集的Y的处理
    train_y_max_len = get_max_len(train_df['Report'])
    train_df['Y'] = train_df['Report'].apply(lambda x: pad_proc(x, train_y_max_len, vocab))

    test_y_max_len = get_max_len(train_df['Report'])
    test_df['Y'] = test_df['Report'].apply(lambda x: pad_proc(x, test_y_max_len, vocab))

    # 保存oov处理后的数据
    train_df['X'].to_csv(config.train_x_pad_path, index=False, header=False)
    train_df['Y'].to_csv(config.train_y_pad_path, index=False, header=False)

    test_df['X'].to_csv(config.test_x_pad_path, index=False, header=False)
    test_df['Y'].to_csv(config.test_y_pad_path, index=False, header=False)

    # oov和pad处理后的数据，词向量重新训练
    print('start retrain word2vec model')
    w2v_model.build_vocab(LineSentence(config.train_x_pad_path), update=True)
    w2v_model.train(LineSentence(config.train_x_pad_path), epochs=config.word2vec_train_epochs, total_examples=w2v_model.corpus_count)

    w2v_model.build_vocab(LineSentence(config.train_y_pad_path), update=True)
    w2v_model.train(LineSentence(config.train_y_pad_path), epochs=config.word2vec_train_epochs, total_examples=w2v_model.corpus_count)

    w2v_model.build_vocab(LineSentence(config.test_x_pad_path), update=True)
    w2v_model.train(LineSentence(config.test_x_pad_path), epochs=config.word2vec_train_epochs, total_examples=w2v_model.corpus_count)

    # 重新保存词向量
    if not path.exists(path.dirname(config.save_w2v_model_path)):
        os.mkdir(path.dirname(config.save_w2v_model_path))
    w2v_model.save(config.save_w2v_model_path)
    print('finish retrain word2vec model .')

    # 更新vocab
    vocab = w2v_model.wv.index_to_key
    print(f'final w2v_model has vocabulary length: {len(vocab)}')

    # 保存到本地
    vocab_key_to_index = w2v_model.wv.key_to_index
    vocab_index_to_key = {index: key for key, index in vocab_key_to_index.items()}
    json.dump(vocab_key_to_index, fp=(open(config.vocab_key_to_index_path, 'w', encoding='utf-8')), ensure_ascii=False)
    json.dump(vocab_index_to_key, fp=(open(config.vocab_index_to_key_path, 'w', encoding='utf-8')), ensure_ascii=False)

    # 保存词向量矩阵
    embedding_matrix = w2v_model.wv.vectors
    np.save(config.embedding_matrix_path, embedding_matrix)

    # 数据集转换，将词转换成索引: [<start> 方向基 ...] -> [2, 403, ...]
    vocab = Vocab()
    train_idx_x = train_df['X'].apply(lambda x: transform_data(x, vocab))
    train_idx_y = train_df['Y'].apply(lambda x: transform_data(x, vocab))

    test_idx_x = train_df['X'].apply(lambda x: transform_data(x, vocab))
    test_idx_y = train_df['Y'].apply(lambda x: transform_data(x, vocab))

    # 数据转换成numpy数组
    train_x = np.array(train_idx_x.tolist())
    train_y = np.array(train_idx_y.tolist())

    test_x = np.array(test_idx_x.tolist())
    test_y = np.array(test_idx_y.tolist())

    # 数据保存
    np.save(config.train_x_path, train_x)
    np.save(config.train_y_path, train_y)

    np.save(config.test_x_path, test_x)
    np.save(config.test_y_path, test_y)

    return train_x, train_y, test_x, test_y


def build_data(train_data_path, test_data_path):

    # 1. train/test data 预处理, 并保存处理后的文件到本地
    train_df, test_df = get_processed_data(train_data_path, test_data_path, is_to_save=True)
    # print(train_df.shape, test_df.shape)

    # 2. 构造train/test用于word2vec训练的数据
    merged_df = get_merged_data(train_df, test_df, is_to_save=True)
    # merged_df.to_csv(config.merged_seg_path, encoding='utf_8_sig', index=False)
    # print(merged_df.head(3))

    # 3. 训练词向量
    w2v_model = train_word2vec(file_path=config.merged_seg_path)
    print(w2v_model)

    # 4. 构造训练与测试的的X, y
    train_x, train_y, test_x, test_y = get_train_test_split(train_df, test_df, w2v_model)


if __name__ == '__main__':
    # 构造数据集
    build_data(train_data_path, test_data_path)