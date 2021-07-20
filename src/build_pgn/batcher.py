# coding=utf-8
import math
import tensorflow as tf

from utils.w2v_helper import Vocab
from utils.params_utils import get_params


def article_to_index(article_words, vocab):

    oov_words = []
    extend_vocab_index = []

    unk_index = vocab.UNKNOWN_TOKEN_INDEX

    for word in article_words:
        word_index = vocab.word_to_index(word)
        if word_index == unk_index:
            if word not in oov_words:
                oov_words.append(word)

            oov_num = oov_words.index(word)
            extend_vocab_index.append(vocab.size() + oov_num)
        else:
            extend_vocab_index.append(word_index)

    return extend_vocab_index, oov_words


def abstract_to_index(abstract_words, vocab, article_oovs):

    index = []
    unk_index = vocab.UNKNOWN_TOKEN_INDEX

    for word in abstract_words:
        word_index = vocab.word_to_index(word)
        if word_index == unk_index:

            if word in article_oovs:
                vocab_index = vocab.size() + article_oovs.index(word)
                index.append(vocab_index)
            else:
                index.append(unk_index)
        else:
            index.append(word_index)

    return index


def get_enc_inp_targ_seqs(sequence, max_len, start_index, stop_index):

    input_index = [start_index] + sequence[:]

    if len(input_index) >= max_len:
        input_index = input_index[: max_len]
    else:
        input_index.append(stop_index)

    return input_index


def get_dec_inp_targ_seqs(sequence, max_len, start_index, stop_index):
    
    input_index = [start_index] + sequence
    target = sequence[:]

    if len(input_index) > max_len:
        input_index = input_index[: max_len]
        target = target[: max_len]
    elif len(input_index) == max_len:
        target.append(stop_index)
    else:
        input_index.append(stop_index)
        target.append(stop_index)
    
    return input_index, target


def example_generator(params, vocab, max_enc_len, max_dec_len, mode):

    if mode != 'test':

        dataset_x = tf.data.TextLineDataset(params[f'{mode}_seg_x_dir'])
        dataset_y = tf.data.TextLineDataset(params[f'{mode}_seg_y_dir'])

        train_dataset = tf.data.Dataset.zip((dataset_x, dataset_y)).take(count=10000)

        if mode == 'train':
            train_dataset = train_dataset.shuffle(10, reshuffle_each_iteration=True).repeat(1)

        for raw_record in train_dataset:

            start_decoding = vocab.word_to_index(vocab.START_DECODING)
            stop_decoding = vocab.word_to_index(vocab.STOP_DECODING)

            article = raw_record[0].numpy().decode('utf-8')
            article_words = article.split()[:max_enc_len]

            enc_input = [vocab.word_to_index(w) for w in article_words]
            enc_input_extend_vocab, article_oovs = article_to_index(article_words, vocab)

            # add start and stop flag
            enc_input = get_enc_inp_targ_seqs(enc_input,
                                              max_enc_len,
                                              start_decoding,
                                              stop_decoding)

            enc_input_extend_vocab = get_enc_inp_targ_seqs(enc_input_extend_vocab,
                                                           max_enc_len,
                                                           start_decoding,
                                                           stop_decoding)

            # mark长度
            enc_len = len(enc_input)
            # 添加mark标记
            encoder_pad_mask = [1 for _ in range(enc_len)]

            abstract = raw_record[1].numpy().decode('utf-8')
            abstract_words = abstract.split()
            abs_ids = [vocab.word_to_index(w) for w in abstract_words]

            dec_input, target = get_dec_inp_targ_seqs(abs_ids,
                                                      max_dec_len,
                                                      start_decoding,
                                                      stop_decoding)

            if params['pointer_gen']:
                abs_ids_extend_vocab = abstract_to_index(abstract_words, vocab, article_oovs)
                _, target = get_dec_inp_targ_seqs(abs_ids_extend_vocab,
                                                  max_dec_len,
                                                  start_decoding,
                                                  stop_decoding)
            # mark长度
            dec_len = len(target)
            # 添加mark标记
            decoder_pad_mask = [1 for _ in range(dec_len)]

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "article_oovs": article_oovs,
                "dec_input": dec_input,
                "target": target,
                "dec_len": dec_len,
                "article": article,
                "abstract": abstract,
                "abstract_sents": abstract,
                "decoder_pad_mask": decoder_pad_mask,
                "encoder_pad_mask": encoder_pad_mask
            }

            yield output
    else:
        test_dataset = tf.data.TextLineDataset(params['valid_seg_x_dir'])
        for raw_record in test_dataset:
            article = raw_record.numpy().decode('utf-8')
            article_words = article.split()[: max_enc_len]
            enc_len = len(article_words)

            enc_input = [vocab.word_to_index(w) for w in article_words]
            enc_input_extend_vocab, article_oovs = article_to_index(article_words, vocab)

            # 添加mark标记
            encoder_pad_mask = [1 for _ in range(enc_len)]

            output = {
                "enc_len": enc_len,
                "enc_input": enc_input,
                "enc_input_extend_vocab": enc_input_extend_vocab,
                "article_oovs": article_oovs,
                "dec_input": [],
                "target": [],
                "dec_len": params['max_dec_len'],
                "article": article,
                "abstract": '',
                "abstract_sents": '',
                "decoder_pad_mask": [],
                "encoder_pad_mask": encoder_pad_mask
            }
            # 每一批的数据都一样阿, 是的是为了beam search
            if params["decode_mode"] == "beam":
                for _ in range(params["batch_size"]):
                    yield output
            elif params["decode_mode"] == "greedy":
                yield output
            else:
                print("shit")


def batch_generator(generator, params, vocab, max_enc_len, max_dec_len, batch_size, mode):
    dataset = tf.data.Dataset.from_generator(lambda: generator(params,
                                                               vocab,
                                                               max_enc_len,
                                                               max_dec_len,
                                                               mode,
                                                               # batch_size
                                                               ),
                                             output_types={
                                                 'enc_len': tf.int32,
                                                 'enc_input': tf.int32,
                                                 'enc_input_extend_vocab': tf.int32,
                                                 'article_oovs': tf.string,
                                                 'dec_input': tf.int32,
                                                 'target': tf.int32,
                                                 'dec_len': tf.int32,
                                                 'article': tf.string,
                                                 'abstract': tf.string,
                                                 'abstract_sents': tf.string,
                                                 'decoder_pad_mask': tf.int32,
                                                 'encoder_pad_mask': tf.int32},
                                             output_shapes={
                                                 'enc_len': [],
                                                 'enc_input': [None],
                                                 'enc_input_extend_vocab': [None],
                                                 'article_oovs': [None],
                                                 'dec_input': [None],
                                                 'target': [None],
                                                 'dec_len': [],
                                                 'article': [],
                                                 'abstract': [],
                                                 'abstract_sents': [],
                                                 'decoder_pad_mask': [None],
                                                 'encoder_pad_mask': [None]})

    dataset = dataset.padded_batch(batch_size=batch_size,
                                   padded_shapes=({'enc_len': [],
                                                   'enc_input': [None],
                                                   'enc_input_extend_vocab': [None],
                                                   'article_oovs': [None],
                                                   'dec_input': [max_dec_len],
                                                   'target': [max_dec_len],
                                                   'dec_len': [],
                                                   'article': [],
                                                   'abstract': [],
                                                   'abstract_sents': [],
                                                   'decoder_pad_mask': [max_dec_len],
                                                   'encoder_pad_mask': [None]}),
                                   padding_values={'enc_len': -1,
                                                   'enc_input': vocab.word2index[vocab.PAD_TOKEN],
                                                   'enc_input_extend_vocab': vocab.word2index[vocab.PAD_TOKEN],
                                                   'article_oovs': b'',
                                                   'dec_input': vocab.word2index[vocab.PAD_TOKEN],
                                                   'target': vocab.word2index[vocab.PAD_TOKEN],
                                                   'dec_len': -1,
                                                   'article': b'',
                                                   'abstract': b'',
                                                   'abstract_sents': b'',
                                                   'decoder_pad_mask': 0,
                                                   'encoder_pad_mask': 0},
                                   drop_remainder=True)

    def update(entry):
        return ({
                    "enc_input": entry["enc_input"],
                    "extended_enc_input": entry["enc_input_extend_vocab"],
                    "article_oovs": entry["article_oovs"],
                    "enc_len": entry["enc_len"],
                    "article": entry["article"],
                    "max_oov_len": tf.shape(entry["article_oovs"])[1],
                    "encoder_pad_mask": entry["encoder_pad_mask"]
                },
                {
                    "dec_input": entry["dec_input"],
                    "dec_target": entry["target"],
                    "dec_len": entry["dec_len"],
                    "abstract": entry["abstract"],
                    "decoder_pad_mask": entry["decoder_pad_mask"]
                })

    dataset = dataset.map(update)

    return dataset


def get_steps_per_epoch(params):

    if params['mode'] == 'train':
        file = open(params['train_seg_y_dir'], 'r', encoding='utf-8')
    elif params['mode'] == 'test':
        file = open(params['test_seg_x_dir'], 'r', encoding='utf-8')
    else:
        file = open(params['valid_seg_x_dir'], 'r', encoding='utf-8')

    num_examples = len(file.readlines())
    if params['decode_mode'] == 'beam':
        return num_examples

    steps_per_epoch = math.ceil(num_examples // params['batch_size'])

    return steps_per_epoch


def batcher(vocab, params):
    dataset = batch_generator(example_generator,
                              params,
                              vocab,
                              params['max_enc_len'],
                              params['max_dec_len'],
                              params['batch_size'],
                              params['mode'],
                              )

    dataset = dataset.prefetch(params['buffer_size'])
    steps_per_epoch = get_steps_per_epoch(params)

    return dataset, steps_per_epoch


if __name__ == '__main__':
    # GPU资源配置
    # config_gpu()
    # 获取参数
    params = get_params()
    params['mode'] = 'train'
    # vocab 对象
    vocab = Vocab()

    b, _ = batcher(vocab, params)

    print(next(iter(b)))