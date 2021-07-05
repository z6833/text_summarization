# coding=utf-8
import math
from tqdm import tqdm
import tensorflow as tf


class Hypothesis:

    def __init__(self, tokens, log_probs, hidden, attn_dists):
        self.tokens = tokens
        self.log_probs = log_probs
        self.hidden = hidden
        self.attn_dists = attn_dists
        self.abstract = ''

    def extend(self, token, log_prob, hidden, attn_dist):

        return Hypothesis(
            tokens=self.tokens + [token],
            log_probs=self.log_probs + [log_prob],
            hidden=hidden,
            attn_dists=self.attn_dists + [attn_dist]
        )

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def total_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.total_log_prob / len(self.tokens)


def batch_greedy_decode(model, batch_data, vocab, params):
    # 判断输入长度
    batch_size = len(batch_data)

    # 存储预测结果
    predictions = [''] * batch_size

    inputs = tf.convert_to_tensor(batch_data)
    # 0. 初始化隐层输入
    init_hidden = tf.zeros(shape=(batch_size, params['enc_units']))
    # 1. 构造encoder
    enc_output, enc_hidden = model.encoder(inputs, init_hidden)

    # 2. 复制到解码器
    dec_hidden = enc_hidden

    # 3. <START> * batch_size
    dec_input = tf.expand_dims([vocab.word_to_index(vocab.START_DECODING)] * batch_size, 1)

    # 4. 解码
    for t in range(params['max_dec_len']):
        # 4.0. 预测
        predictions, dec_hidden, attention_weights = model.decoder(dec_input, dec_hidden, enc_output)

        # 4.1. 取预测结果，概率最大值所对应的index
        predictions_idx = tf.argmax(predictions, axis=1).numpy()  # 最大值所对应的角标

        # 4.2. 根据index，取相应的词,存放到列表
        for index, predict_idx in enumerate(predictions_idx):
            predictions[index] += vocab.index_to_word(predict_idx) + ' '

        # 4.3. 继续下一个词的预测（用上一步预测的结果）
        dec_input = tf.expand_dims(predictions_idx)

    # 5. 解码结果处理
    results = []
    for prediction in predictions:

        prediction = prediction.strip()
        if vocab.STOP_DECODING in prediction:
            prediction = prediction[:prediction.index(vocab.STOP_DECODING)]
        results.append(prediction)

    return results


def greedy_decode(model, test_x, batch_size, vocab, params):
    # 存储结果
    results = []
    samples_size = len(test_x)

    # 一共可以取多少个batch(向上取整)
    steps_epoch = math.ceil(samples_size / batch_size)
    for i in tqdm(range(steps_epoch)):
        batch_data = test_x[i * batch_size: (i + 1) * batch_size]
        results += batch_greedy_decode(model, batch_data, vocab, params)

    return results


def print_top_k(hyp, k, vocab, batch_data):

    text = ' '.join([vocab.index_to_word(int(index)) for index in batch_data[0]])
    print(f'hyp.text: {text}')
    for i in range(min(k, len(hyp))):
        k_hyp = hyp[i]
        k_hyp.abstract = ' '.join([vocab.index_to_word(int(index)) for index in k_hyp.tokens])
        print(f'top {i} best_hyp.abstract: {k_hyp.abstract}')


def beam_decode(model, batch_data, vocab, params):
    # 初始化mask
    start_index = vocab.STOP_DECODING_INDEX
    stop_index = vocab.STOP_DECODING_INDEX
    unk_index = vocab.UNKNOWN_TOKEN_INDEX
    batch_size = params['batch_size']

    # 单步decoder
    def decoder_one_step(enc_output, dec_input, dec_hidden):
        final_pred, dec_hidden, attention_weights = model.decoder(dec_input, dec_hidden, enc_output)

        # 取top K个index及其对应的概率
        top_k_probs, top_k_idx = tf.nn.top_k(tf.squeeze(final_pred), k=params['beam_size'] * 2)

        # 重新计算概率分布
        top_k_log_probs = tf.math.log(top_k_probs)

        results = {
            'dec_hidden': dec_hidden,
            'attention_weights': attention_weights,
            'top_k_idx': top_k_idx,
            'top_k_log_probs': top_k_log_probs
        }

        return results

    # 测试数据的输入
    enc_input = batch_data
    init_enc_hidden = model.encoder.initialize_hidden_state()

    # 计算encoder的输出
    enc_output, enc_hidden = model.encoder(enc_input, init_enc_hidden)

    hyps_batch = [Hypothesis(tokens=[start_index],
                             log_probs=[0.],
                             hidden=enc_hidden[0],
                             attn_dists=[]) for _ in range(batch_size)]

    # 初始化结果集合
    results = []
    steps = 0  # 遍历步数

    # 当长度不够或者结果还不够时，继续搜索
    while steps < params['max_dec_len'] and len(results) < params['beam_size']:

        # 获取最新待使用的token
        latest_tokens = [hyps.latest_token for hyps in hyps_batch]
        # 替换掉oov token为unk token
        latest_tokens = [token if token in vocab.index2word else unk_index for token in latest_tokens]

        # 获取隐变量
        hiddens = [hyps.hidden for hyps in hyps_batch]

        dec_input = tf.expand_dims(latest_tokens, axis=1)
        dec_hidden = tf.stack(hiddens, axis=0)

        # 单步运行decoder
        decoder_results = decoder_one_step(enc_output, dec_input, dec_hidden)

        dec_hidden = decoder_results['dec_hidden']
        attention_weights = decoder_results['attention_weights']
        top_k_log_probs = decoder_results['top_k_log_probs']
        top_k_idx = decoder_results['top_k_idx']

        # 现阶段全部可能的情况
        all_hyps = []

        # 原有的所有可能情况
        num_ori_hyps = 1 if steps == 0 else len(hyps_batch)

        # 便利添加所有可能的结果
        for i in range(num_ori_hyps):
            hyps, new_hidden, attn_dist = hyps_batch[i], dec_hidden[i], attention_weights[i]

            for j in range(params['beam_size'] * 2):
                new_hyps = hyps.extend(
                    token=top_k_idx[i, j].numpy(),
                    log_prob=top_k_log_probs[i, j],
                    hidden=new_hidden,
                    attn_dist=attn_dist
                )

                all_hyps.append(new_hyps)


        # 重置
        hyps_batch = []
        sorted_hyps = sorted(all_hyps, key=lambda h: h.ave_log_prob, reverse=True)

        # 筛选
        for h in sorted_hyps:
            if h.latest_token == stop_index:
                # 长度符合预测，遇到居委，添加到结果集
                if steps >= params['min_dec_steps']:
                    h.tokens = h.tokens[1: -1]
                    results.append(h)

            else:
                hyps.append(h)

            if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                break

        steps += 1

    if len(results) == 0:
        results = hyps

    hyps_sorted = sorted(results, key=lambda h: h.ave_log_prob, reverse=True)
    print_top_k(hyps_sorted, 3, vocab, batch_data)

    best_hyp = hyps_sorted[0]
    best_hyp.abstract = ' '.join([vocab.index_to_word(index) for index in best_hyp.tokens])

    return best_hyp
