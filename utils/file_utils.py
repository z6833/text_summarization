# coding=utf-8
import time
import os

from utils.config import results_dir


def get_result_filename(batch_size, epochs, max_length_inp, embedding_dim, commit=''):
    """获取时间
    :return:
    """
    now_time = time.strftime('%Y_%m_%d_%H_%M_%S')
    filename = now_time + f'_batch_size_{batch_size}_epochs_{epochs}_max_length_inp_{max_length_inp}_embedding_dim_{embedding_dim}{commit}.csv'

    # result_save_path = os.path.join(results_dir, filename)
    return filename
