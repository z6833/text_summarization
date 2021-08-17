import numpy as np
import tensorflow as tf

from src.build_seq2seq_transformer_pgn.layers.common import point_wise_feed_forward_network


def get_angles(pos, i, d_model):

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # 将 sin 应用于数组中的偶数索引（indices）；2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # 将 cos 应用于数组中的奇数索引；2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


if __name__ == '__main__':
    sample_ffn = point_wise_feed_forward_network(512, 2048)
    print(sample_ffn(tf.random.uniform((64, 50, 512))).shape)
