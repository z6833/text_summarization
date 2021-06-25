# coding=utf-8
import json
import numpy as np
from gensim.models import Word2Vec

from utils import config


class Vocab:

    PAD_TOKEN = '<PAD>'
    UNKNOWN_TOKEN = '<UNK>'
    START_DECODING = '<START>'
    STOP_DECODING = '<STOP>'

    MASKS = [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]
    MASKS_COUNT = len(MASKS)

    PAD_TOKEN_INDEX = MASKS.index(PAD_TOKEN)
    UNKNOWN_TOKEN_INDEX = MASKS.index(UNKNOWN_TOKEN)
    START_DECODING_INDEX = MASKS.index(START_DECODING)
    STOP_DECODING_INDEX = MASKS.index(STOP_DECODING)

    def __init__(self, vocab_file=config.vocab_key_to_index_path, vocab_max_size=None):
        """
        vocab基本操作封装
        :param vocab_file:
        :param vocab_max_size:
        """

        self.word2index, self.index2word = self.load_vocab(vocab_file, vocab_max_size)
        self.count = len(self.word2index)

    @staticmethod
    def load_vocab(file_path, vocab_max_size=None):

        word2index = {mask: index for index, mask in enumerate(Vocab.MASKS)}
        index2word = {index: mask for index, mask in enumerate(Vocab.MASKS)}

        vocab_dict = list(json.load(fp=open(file_path, 'r', encoding='utf-8')).items())[:-4]
        vocab_dict = vocab_dict if vocab_max_size is None else vocab_dict[: vocab_max_size]

        for word, index in vocab_dict:
            word2index[word] = index + Vocab.MASKS_COUNT
            index2word[index + Vocab.MASKS_COUNT] = word

        return word2index, index2word

    def word_to_index(self, word):

        return self.word2index[word] if word in self.word2index else self.word2index[self.UNKNOWN_TOKEN]

    def index_to_word(self, word_index):

        assert word_index in self.index2word, f'word index [{word_index}] not found in vocab'

        return self.index2word[word_index]

    def size(self):
        return self.count


def load_embedding_matrix(file_path=config.embedding_matrix_path, max_vocab_size=102400):
    embedding_matrix = np.load(file_path + '.npy')
    flag_matrix = np.zeros_like(embedding_matrix[:Vocab.MASKS_COUNT])
    return np.concatenate([flag_matrix, embedding_matrix])[: max_vocab_size]


def load_word2vec_model():

    return Word2Vec.load(config.save_w2v_model_path)


if __name__ == "__main__":

    vocab = Vocab()
    # vocab.load_vocab(config.vocab_key_to_index_path)
    #
    # print(vocab.size())
    print(vocab.word_to_index('<PAD>'))
    print(vocab.word2index['<PAD>'])
    # print(vocab.index_to_word(1024))
