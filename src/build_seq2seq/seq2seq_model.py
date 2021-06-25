# coding=utf-8

import tensorflow as tf
from tensorflow import keras

from utils.w2v_helper import load_embedding_matrix
from model_layers import Encoder, Decoder, BahdanauAttention


class Seq2Seq(keras.Model):

    def __init__(self, params, vocab):
        super(Seq2Seq, self).__init__()

        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        self.vocab = vocab

        self.batch_size = params['batch_size']

        self.enc_units = params['enc_units']
        self.dec_units = params['dec_units']
        self.att_units = params['att_units']

        self.encoder = Encoder(self.embedding_matrix, self.enc_units, self.batch_size)
        self.decoder = Decoder(self.embedding_matrix, self.dec_units, self.batch_size)

        self.attention = BahdanauAttention(self.att_units)

    def teacher_decoder(self, dec_hidden, enc_output, dec_target):

        prediction = []

        # 第一个输入<START>
        dec_input = tf.expand_dims([self.vocab.START_DECODING_INDEX] * self.batch_size, axis=1)

        # teacher forcing 讲target作为下一次的输入，依次解码
        for t in range(1, dec_target.shape[1]):  # dec_target shape: batch_size * max_len
            pred, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)

            # 预测下一个值需要的输入
            dec_input = tf.expand_dims(dec_target[:, t], axis=1)

            prediction.append(pred)

        return tf.stack(prediction, axis=1), dec_hidden





