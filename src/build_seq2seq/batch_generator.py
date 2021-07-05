# coding=utf-8
import tensorflow as tf
from tqdm import tqdm

from utils import config
from utils.data_loader import load_dataset


def beam_test_batch_generator(beam_size, max_enc_len=200, max_dec_len=50):
    test_x, _ = load_dataset(config.test_x_path, config.test_y_path, max_enc_len, max_dec_len)

    print(f'total test samples: {len(test_x)} .')
    for row in tqdm(test_x, total=len(test_x), desc='beam search'):

        beam_search_data = tf.convert_to_tensor([row for _ in range(beam_size)])
        yield beam_search_data




def train_batch_generator(batch_size, max_enc_len=200, max_dec_len=50, sample_sum=None):

    train_X, train_Y = load_dataset(config.train_x_path,
                                    config.train_y_path,
                                    max_enc_len,
                                    max_dec_len,
                                    sample_sum)

    valid_X, valid_Y = load_dataset(config.test_x_path,
                                    config.test_y_path,
                                    max_enc_len,
                                    max_dec_len,
                                    sample_sum)

    print(f'total samples: {len(train_Y)} for training ...')

    train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_X, valid_Y))

    train_dataset = train_dataset.shuffle(len(train_X), reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)
    valid_dataset = valid_dataset.shuffle(len(valid_X), reshuffle_each_iteration=True).batch(batch_size, drop_remainder=True)

    train_steps_per_epoch = len(train_X) // batch_size
    valid_steps_per_epoch = len(valid_X) // batch_size

    return train_dataset, valid_dataset, train_steps_per_epoch, valid_steps_per_epoch
