# coding=utf-8
import tensorflow as tf
from tensorflow import keras

from utils.w2v_helper import Vocab, load_embedding_matrix

class Encoder(keras.Model):

    def __init__(self, embedding_matrix, enc_units, batch_size):
        super(Encoder, self).__init__()

        self.batch_size = batch_size
        self.enc_units = enc_units
        self.vocab_size, self.embedding_dim = embedding_matrix.shape

        self.embedding = keras.layers.Embedding(self.vocab_size,
                                                self.embedding_dim,
                                                weights=[embedding_matrix],
                                                trainable=False)

        self.gru = keras.layers.GRU(self.enc_units,
                                    return_state=True,
                                    return_sequences=True,
                                    recurrent_initializer='glorot_uniform')

        self.bidirectional_gru = keras.layers.Bidirectional(self.gru)

    def call(self, x, enc_hidden):

        x = self.embedding(x)  # x shape: batch_size * enc_units -> batch_size * 128

        # enc_output shape: batch * max_len * enc_unit
        enc_output, forward_state, backward_state = self.bidirectional_gru(x, initial_state=[enc_hidden, enc_hidden])

        # enc_hidden shape: batch_size * 256
        enc_hidden = keras.layers.concatenate([forward_state, backward_state], axis=-1)

        return enc_output, enc_hidden

    def initialize_hidden_state(self):

        return tf.zeros(shape=(self.batch_size, self.enc_units))


def masked_attention(enc_pad_mask, attn_dist):

    attn_dist = tf.squeeze(attn_dist, axis=2)
    mask = tf.cast(enc_pad_mask, dtype=attn_dist.dtype)

    attn_dist *= mask

    mask_sum = tf.reduce_sum(attn_dist, axis=1)
    attn_dist = attn_dist / tf.reshape(mask_sum + 1e-12, [-1, 1])

    attn_dist = tf.expand_dims(attn_dist, axis=2)

    return attn_dist


class BahdanauAttention(keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        self.W_s = keras.layers.Dense(units)
        self.W_h = keras.layers.Dense(units)
        self.W_c = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, dec_hidden, enc_output, enc_pad_mask, use_coverage=False, pre_coverage=None):

        hidden_with_time_axis = tf.expand_dims(dec_hidden, 1)

        if use_coverage and pre_coverage is not None:
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis) + self.W_c(pre_coverage)))

            attention_weights = tf.nn.softmax(score, axis=1)
            attention_weights = masked_attention(enc_pad_mask, attention_weights)
            coverage = attention_weights + pre_coverage
        else:
            score = self.V(tf.nn.tanh(self.W_s(enc_output) + self.W_h(hidden_with_time_axis)))

            attention_weights = tf.nn.softmax(score)
            attention_weights = masked_attention(enc_pad_mask, attention_weights)

            if use_coverage:
                coverage = attention_weights
            else:
                coverage = []

        context_vactor = attention_weights * enc_output
        context_vactor = tf.reduce_sum(context_vactor, axis=1)

        return context_vactor, tf.squeeze(attention_weights, -1), coverage


class Decoder(keras.Model):

    def __init__(self, embedding_matrix, dec_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.vocab_size, self.embedding_dim = embedding_matrix.shape

        self.embedding = keras.layers.Embedding(self.vocab_size,
                                                self.embedding_dim,
                                                weights=[embedding_matrix],
                                                trainable=False)
        self.cell = keras.layers.GRUCell(units=self.dec_units, recurrent_initializer='glorot_uniform')

        self.fc = keras.layers.Dense(self.vocab_size, activation=keras.activations.softmax)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, dec_input, dec_hidden, enc_output, enc_pad_mask, pre_coverage, use_covarage=True):

        dec_x = self.embedding(dec_input)

        dec_output, [dec_hidden] = self.cell(dec_x, [dec_hidden])

        context_vector, attention_weights, coverage = self.attention(dec_hidden,
                                                                     enc_output,
                                                                     enc_pad_mask,
                                                                     use_covarage,
                                                                     pre_coverage)

        dec_output = tf.concat([dec_output, context_vector], axis=-1)
        prediction = self.fc(dec_output)

        return context_vector, dec_hidden, dec_x, prediction, attention_weights, coverage


class Pointer(keras.layers.Layer):

    def __init__(self):
        super(Pointer, self).__init__()

        self.w_s_reduce = keras.layers.Dense(1)
        self.w_i_reduce = keras.layers.Dense(1)
        self.w_c_reduce = keras.layers.Dense(1)

    def call(self, context_vector, dec_hidden, dec_inp):

        return tf.nn.sigmoid(self.w_s_reduce(dec_hidden) +
                             self.w_c_reduce(context_vector) +
                             self.w_i_reduce(dec_inp))


if __name__ == '__main__':
    # GPU资源配置
    # config_gpu()
    # 读取vocab训练
    vocab = Vocab()
    # 计算vocab size
    vocab_size = vocab.count

    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()

    enc_max_len = 200
    dec_max_len = 41
    batch_size = 64
    embedding_dim = 300
    enc_units = 512
    dec_units = 1024
    att_units = 20

    # 编码器结构
    encoder = Encoder(embedding_matrix, enc_units, batch_size)
    # encoder input
    enc_inp = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)
    # decoder input
    dec_inp = tf.ones(shape=(batch_size, dec_max_len), dtype=tf.int32)
    # enc pad mask
    enc_pad_mask = tf.ones(shape=(batch_size, enc_max_len), dtype=tf.int32)

    # encoder hidden
    enc_hidden = encoder.initialize_hidden_state()
    # encoder hidden
    enc_output, enc_hidden = encoder(enc_inp, enc_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, units) {}'.format(enc_output.shape))
    print('Encoder Hidden state shape: (batch size, units) {}'.format(enc_hidden.shape))

    # dec_hidden, enc_output, enc_pad_mask, use_coverage = False, prev_coverage = None)

    dec_hidden = enc_hidden
    attention_layer = BahdanauAttention(att_units)
    context_vector, attention_weights, coverage = attention_layer(dec_hidden, enc_output, enc_pad_mask,
                                                                  use_coverage=True, prev_coverage=None)

    print("Attention context_vector shape: (batch size, units) {}".format(context_vector.shape))
    print("Attention weights shape: (batch_size, sequence_length) {}".format(attention_weights.shape))
    print("Attention coverage: (batch_size,sequence_length ) {}".format(coverage.shape))

    decoder = Decoder(embedding_matrix, dec_units, batch_size)

    prev_dec_hidden = enc_hidden
    prev_coverage = coverage

    context_vector, dec_hidden, dec_x, prediction, attention_weights, coverage = decoder(dec_inp[:, 0],
                                                                                         prev_dec_hidden,
                                                                                         enc_output,
                                                                                         enc_pad_mask,
                                                                                         prev_coverage,
                                                                                         use_coverage=True)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(prediction.shape))
    print('Decoder dec_x shape: (batch_size, embedding_dim) {}'.format(dec_x.shape))
    print('Decoder context_vector shape: (batch_size, 1,dec_units) {}'.format(context_vector.shape))
    print('Decoder attention_weights shape: (batch_size, sequence_length) {}'.format(attention_weights.shape))
    print('Decoder dec_hidden shape: (batch_size, dec_units) {}'.format(dec_hidden.shape))

    pointer = Pointer()
    p_gen = pointer(context_vector, dec_hidden, dec_x)
    print('Pointer p_gen shape: (batch_size,1) {}'.format(p_gen.shape))
