import tensorflow as tf
import numpy as np
from tqdm import tqdm

from src.build_seq2seq_transformer.layers.transformer import create_masks


class Hypothesis:
    """ Class designed to hold hypothesises throughout the beamSearch decoding """

    def __init__(self, tokens, log_probs, attn_dists):
        # list of all the tokens from time 0 to the current time step t
        self.tokens = tokens
        # list of the log probabilities of the tokens of the tokens
        self.log_probs = log_probs
        # attention dists of all the tokens
        self.attn_dists = attn_dists
        # generation probability of all the tokens

    def extend(self, token, log_prob, attn_dist):
        """Method to extend the current hypothesis by adding the next decoded token and all
        the informations associated with it"""
        return Hypothesis(tokens=self.tokens + [token],  # we add the decoded token
                          log_probs=self.log_probs + [log_prob],  # we add the log prob of the decoded token
                          attn_dists=self.attn_dists + [attn_dist])

    @property
    def latest_token(self):
        return self.tokens[-1]

    @property
    def tot_log_prob(self):
        return sum(self.log_probs)

    @property
    def avg_log_prob(self):
        return self.tot_log_prob / len(self.tokens)


def beam_decode(model, batch, vocab, params, print_info=False):
    def decode_onestep(enc_inp, enc_extended_inp, dec_inp, batch_oov_len):
        """
            Method to decode the output step by step (used for beamSearch decoding)
            Args:
                sess : tf.Session object
                batch : current batch, shape = [beam_size, 1, vocab_size( + max_oov_len if pointer_gen)]
                (for the beam search decoding, batch_size = beam_size)
                enc_outputs : hiddens outputs computed by the encoder LSTM
                dec_state : beam_size-many list of decoder previous state, LSTMStateTuple objects,
                shape = [beam_size, 2, hidden_size]
                dec_input : decoder_input, the previous decoded batch_size-many words, shape = [beam_size, embed_size]
                cov_vec : beam_size-many list of previous coverage vector
            Returns: A dictionary of the results of all the ops computations (see below for more details)
        """
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(enc_inp, dec_inp)
        outputs = model(enc_inp,
                        enc_extended_inp,
                        batch_oov_len,
                        dec_inp,
                        params['training'],
                        enc_padding_mask,
                        combined_mask,
                        dec_padding_mask)
        final_dists = outputs["logits"]
        attentions = outputs["attentions"]

        # final_dists shape=(3, 1, 30000)
        # top_k_probs shape=(3, 6)
        # top_k_ids shape=(3, 6)
        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(final_dists), k=params["beam_size"] * 2)
        top_k_log_probs = tf.math.log(top_k_probs)

        # dec_hidden shape = (3, 256)
        # attentions, shape = (3, 115)
        # p_gens shape = (3, 1)
        # coverages,shape = (3, 115, 1)
        results = {"attention_vec": attentions,  # [batch_sz, max_len_x, 1]
                   "top_k_ids": top_k_ids,
                   "top_k_log_probs": top_k_log_probs}
        return results

    # end of the nested class
    # We run the encoder once and then we use the results to decode each time step token
    # state shape=(3, 256), enc_outputs shape=(3, 115, 256)
    # enc_input = batch[0]["enc_input"]
    # enc_outputs, state = model.call_encoder(enc_input)
    # Initial Hypothesises (beam_size many list)
    hyps = [Hypothesis(tokens=[vocab.START_DECODING_INDEX],
                       log_probs=[0.0],
                       attn_dists=[]) for _ in range(params['batch_size'])]

    results = []  # list to hold the top beam_size hypothesises
    steps = 0  # initial step
    while steps < params['max_dec_steps'] and len(results) < params['beam_size']:
        latest_tokens = [h.latest_token for h in hyps]  # latest token for each hypothesis , shape : [beam_size]
        # we replace all the oov is by the unknown token
        latest_tokens = [t if t in range(params['vocab_size']) else vocab.UNKNOWN_TOKEN_INDEX for t in latest_tokens]
        # we collect the last states for each hypothesis

        # we decode the top likely 2 x beam_size tokens tokens at time step t for each hypothesis
        # model, batch, vocab, params
        dec_input = tf.expand_dims(latest_tokens, axis=1)  # shape=(3, 1)
        returns = decode_onestep(batch[0]['enc_input'],  # shape=(3, 115)
                                 batch[0]['extended_enc_input'],  # shape=(3, 115)
                                 dec_input,  # shape=(3, 1)
                                 batch[0]["max_oov_len"])  # shape=(3, 115, 1)

        topk_ids, topk_log_probs, attn_dists = returns['top_k_ids'], \
                                               returns['top_k_log_probs'], \
                                               returns['attention_vec']

        all_hyps = []
        num_orig_hyps = 1 if steps == 0 else len(hyps)
        for i in range(num_orig_hyps):
            h = hyps[i]
            attn_dist = attn_dists[i]
            for j in range(params['beam_size'] * 2):
                # we extend each hypothesis with each of the top k tokens
                # (this gives 2 x beam_size new hypothesises for each of the beam_size old hypothesises)
                new_hyp = h.extend(token=topk_ids[i, j].numpy(),
                                   log_prob=topk_log_probs[i, j],
                                   attn_dist=attn_dist)
                all_hyps.append(new_hyp)

        # in the following lines, we sort all the hypothesises, and select only the beam_size most likely hypothesises
        hyps = []
        sorted_hyps = sorted(all_hyps, key=lambda h: h.avg_log_prob, reverse=True)
        for h in sorted_hyps:
            if h.latest_token == vocab.STOP_DECODING_INDEX:
                if steps >= params['min_dec_steps']:
                    results.append(h)
            else:
                hyps.append(h)

            # ??????????????????????????????beam_size ???????????????????????????beam_size ???????????????
            if len(hyps) == params['beam_size'] or len(results) == params['beam_size']:
                break
        steps += 1

    if len(results) == 0:
        results = hyps

    # At the end of the loop we return the most likely hypothesis, which holds the most likely ouput sequence,
    # given the input fed to the model
    hyps_sorted = sorted(results, key=lambda h: h.avg_log_prob, reverse=True)

    best_hyp = hyps_sorted[0]
    best_hyp = result_index2text(best_hyp, vocab, batch)

    if print_info:
        print_top_k(hyps_sorted, params['beam_size'], vocab, batch)
        print('real_article: {}'.format(best_hyp.real_abstract))
        print('article: {}'.format(best_hyp.abstract))

    return best_hyp


def print_top_k(hyp, k, vocab, batch):
    text = batch[0]["article"].numpy()[0].decode()

    print('\nhyp.text :{}'.format(text))
    for i in range(min(k, len(hyp))):
        k_hyp = hyp[i]
        k_hyp = result_index2text(k_hyp, vocab, batch)
        print('top {} best_hyp.abstract :{}\n'.format(i, k_hyp.abstract))


def result_index2text(hyp, vocab, batch):
    article_oovs = batch[0]["article_oovs"].numpy()[0]
    hyp.real_abstract = batch[1]["abstract"].numpy()[0].decode()
    hyp.article = batch[0]["article"].numpy()[0].decode()

    words = []
    for index in hyp.tokens:
        if index != 2 and index != 3:
            if index < (len(article_oovs) + vocab.size()):
                if index < vocab.size():
                    words.append(vocab.id_to_word(index))
                else:
                    words.append(article_oovs[index - vocab.size()].decode())
            else:
                print('error values id :{}'.format(index))
    hyp.abstract = " ".join(words)
    return hyp


def batch_greedy_decode(model, encoder_batch_data, vocab, params):
    # ??????????????????
    enc_input = encoder_batch_data["enc_input"]
    enc_extended_inp = encoder_batch_data["extended_enc_input"]
    batch_size = encoder_batch_data["enc_input"].shape[0]

    article_oovs = encoder_batch_data["article_oovs"]
    # ??????????????????list
    predicts = [''] * batch_size

    dec_input = tf.constant([vocab.START_DECODING_INDEX] * batch_size)
    dec_input = tf.expand_dims(dec_input, axis=1)
    # Teacher forcing - feeding the target as the next input

    try:
        batch_oov_len = tf.shape(encoder_batch_data["article_oovs"])[1]
    except:
        batch_oov_len = tf.constant(0)

    for t in range(params['max_dec_len']):
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(enc_input, dec_input)

        # ????????????
        outputs = model(enc_input,
                        enc_extended_inp,
                        batch_oov_len,
                        dec_input,
                        params['training'],
                        enc_padding_mask,
                        combined_mask,
                        dec_padding_mask)

        final_dists = outputs["logits"][:, -1, :]

        # id??????
        predicted_ids = tf.argmax(final_dists, axis=-1)

        inp_predicts = []
        for index, predicted_id in enumerate(predicted_ids.numpy()):
            if predicted_id >= vocab.count:
                # OOV???
                word = article_oovs[index][int(predicted_id) - vocab.count].numpy().decode()
                inp_predicts.append(vocab.UNKNOWN_TOKEN_INDEX)
            else:
                word = vocab.id_to_word(int(predicted_id))
                inp_predicts.append(int(predicted_id))

            predicts[index] += word + ' '

        predicted_ids = np.array(inp_predicts, dtype=np.int32)
        predicted_ids = tf.expand_dims(predicted_ids, axis=1)

        # using teacher forcing
        dec_input = tf.concat([dec_input, predicted_ids], axis=-1)

    results = []
    for predict in predicts:
        # ????????????????????????
        predict = predict.strip()
        # ????????????max len???????????? ??????
        if vocab.STOP_DECODING in predict:
            # ??????stop
            predict = predict[:predict.index(vocab.STOP_DECODING)]
        # ????????????
        results.append(predict)
    return results


def greedy_decode(model, dataset, vocab, params):

    # ????????????
    results = []

    for enc_data, _ in tqdm(dataset, total=params['steps_per_epoch']):
        batch_results = batch_greedy_decode(model, enc_data, vocab, params)
        print(batch_results[0])
        results += batch_results

    return results
