# coding=utf-8
import numpy as np


def load_dataset(x_path, y_path, max_enc_len, max_dec_len, samlpe_sum=None):
    X = np.load(x_path + '.npy')
    Y = np.load(y_path + '.npy')

    if samlpe_sum:
        X = X[:samlpe_sum, :max_enc_len]
        Y = Y[:samlpe_sum, :max_dec_len]
    else:
        X = X[:, :max_enc_len]
        Y = Y[:, :max_dec_len]

    return X, Y