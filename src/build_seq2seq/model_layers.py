# coding=utf-8

import tensorflow as tf
from tensorflow import keras

from utils.params_utils import get_params
from utils.w2v_helper import Vocab, load_embedding_matrix

class Encoder(keras.Model):

    def __init__(self, embedding_matrix, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = keras.layers.Embedding(vocab_size,
                                                embedding_dim,
                                                weights=[embedding_matrix],
                                                trainable=False)

        self.gru = keras.layers.GRU(units=self.enc_units,
                                    return_state=True,
                                    return_sequences=True,
                                    recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):

        # embedding前x维度：batch_size * max_len -> 32 * 341
        x = self.embedding(x)
        # embedding后x维度：batch_size * max_len * embedding_dim -> 32 * 341 * 300

        # output 维度：batch_size * max_len * enc_units  -> 32 * 341 * 400
        # state  维度：batch_size * enc_units  -> 32 * 400
        output, state = self.gru(x, initial_state=hidden)

        return output, state

    def initialize_hidden_state(self):

        return tf.zeros(shape=(self.batch_sz, self.enc_units))


class BahdanauAttention(keras.Model):

    def __init__(self, units):
        super(BahdanauAttention, self).__init__()

        self.W1 = keras.layers.Dense(units)
        self.W2 = keras.layers.Dense(units)
        self.V = keras.layers.Dense(1)

    def call(self, query, values):

        # query为decoder中，上一个时间步的隐变量St-1
        # values为encoder的编码结果enc_output
        # seq2seq模型中，st是decoder中的query向量;而encoder的隐变量hi是values

        # query 维度：batch_size * dec_units -> 32 * 400
        # values维度：batch_size * max_len * dec_units -> 32 * 341 * 400

        # hidden_with_time_axis维度：batch_size * 1 * dec_units
        hidden_with_time_axis = tf.expand_dims(query, axis=1)

        # self.W1(values): batch_size * max_len * dec_units
        # self.W2(hidden_with_time_axis)： batch_size * 1 * dec_units
        # tanh(...)维度：batch_size * max_len * dec_units  tf加法性质：对应相加

        # score维度：batch_size * max_len * 1 -> 32 * 341 * 1
        score = self.V(
            tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis))
        )

        # attention_weights维度：batch_size * max_len  * 1
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector维度：batch_size * dec_units -> 32 * 400
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(keras.Model):

    def __init__(self, embedding_matrix, dec_units, batch_sz):
        super(Decoder, self).__init__()

        self.batch_sz = batch_sz
        self.dec_units = dec_units
        vocab_size, embedding_dim = embedding_matrix.shape

        self.embedding = keras.layers.Embedding(vocab_size,
                                                embedding_dim,
                                                weights=[embedding_matrix],
                                                trainable=False)

        self.gru = keras.layers.GRU(self.dec_units,
                                    return_state=True,
                                    return_sequences=True,
                                    recurrent_initializer='glorot_uniform')

        self.fc = keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):

        # hidden维度：batch_size * dec_units  -> 32 * 400
        # enc_output维度：batch_size * max_len * dec_units -> 32 * 341 * 400
        # context_vector维度：batch_size * dec_units -> 32 * 400
        # attention_weights维度：batch_size * max_len  * 1
        context_vector, attention_weight = self.attention(hidden, enc_output)

        # embedding后x维度：batch_size * 1 * embedding_dim-> 32 * 1 * 300
        x = self.embedding(x)

        # x拼接后的维度：batch_size * 1 * dec_units + embedding_dim -> 32 * 341 * (400 + 300)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # output维度：batch_size * 1 * 400
        # state 维度：batch_size * 400
        output, state = self.gru(x, hidden)

        # output维度：batch_size * 400
        output = tf.reshape(output, shape=(-1, output.shape[2]))

        # prediction维度：batch_size * len(vocab)
        prediction = self.fc(output)

        return prediction, state, attention_weight


if __name__ == '__main__':
    # GPU资源配置
    # config_gpu()
    # 获得参数
    params = get_params()
    # 读取vocab训练
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    # 计算vocab size
    vocab_size = vocab.count
    # 使用GenSim训练好的embedding matrix
    embedding_matrix = load_embedding_matrix()

    input_sequence_len = 250
    batch_size = 64
    embedding_dim = 500
    units = 1024

    # 编码器结构 embedding_matrix, enc_units, batch_sz
    encoder = Encoder(embedding_matrix, units, batch_size)
    # example_input
    example_input_batch = tf.ones(shape=(batch_size, input_sequence_len), dtype=tf.int32)
    # sample input
    sample_hidden = encoder.initialize_hidden_state()

    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)
    # 打印结果
    print('Encoder output shape: (batch size, sequence length, hidden_dim) {}'.format(sample_output.shape))
    print('Encoder Hidden state shape: (batch size, hidden_dim) {}'.format(sample_hidden.shape))

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
    print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))

    decoder = Decoder(embedding_matrix, units, batch_size)
    sample_decoder_output, state, attention_weights = decoder(tf.random.uniform((64, 1)),
                                                              sample_hidden, sample_output)

    print('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))