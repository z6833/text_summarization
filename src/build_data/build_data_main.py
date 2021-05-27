# coding=utf-8
import pandas as pd

from utils.config import train_data_path, test_data_path
from utils import config
from preprocess import pre_process
from tools import save_to_csv
from train_vector import train_word2vec


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

if __name__ == '__main__':
    # 构造数据集
    build_data(train_data_path, test_data_path)