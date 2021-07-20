# coding=utf-8
import tensorflow as tf
from tensorflow import keras

from utils.w2v_helper import load_embedding_matrix
from src.build_pgn.layers import Encoder, Decoder, Pointer


class PGN(keras.Model):

    def __init__(self, params):
        super(PGN, self).__init__()
        self.embedding_matrix = load_embedding_matrix(max_vocab_size=params['vocab_size'])

        self.vocab_size = params['vocab_size']
        self.batch_size = params['batch_size']

        self.encoder = Encoder(self.embedding_matrix,
                               params['enc_units'],
                               params['batch_size'])

        self.decoder = Decoder(self.embedding_matrix,
                               params['dec_units'],
                               params['batch_size'])

        self.pointer = Pointer()

    def call_one_step(self, dec_input, dec_hidden, enc_output, enc_pad_mask, use_coverage, prev_coverage):
        context_vector, dec_hidden, dec_x, prediction, attention_weights, coverage = self.decoder(dec_input,
                                                                                                  dec_hidden,
                                                                                                  enc_output,
                                                                                                  enc_pad_mask,
                                                                                                  prev_coverage,
                                                                                                  use_coverage)

        p_gens = self.pointer(context_vector, dec_hidden, dec_x)

        return prediction, dec_hidden, context_vector, attention_weights, p_gens, coverage

    def call(self, dec_input, dec_hidden, enc_output, enc_extended_input, batch_oov_len, enc_pad_mask, use_coverage,
             coverage=None):
        predictions = []
        attentions = []
        p_gens = []
        coverages = []

        for t in range(dec_input.shape[1]):
            final_dists, dec_hidden, context_vector, attention_weights, p_gen, coverage = self.call_one_step(
                dec_input[:, t],
                dec_hidden,
                enc_output,
                enc_pad_mask,
                use_coverage,
                coverage)

            coverages.append(coverage)
            predictions.append(final_dists)
            attentions.append(attention_weights)
            p_gens.append(p_gen)

        final_dists = _calc_final_dist(enc_extended_input,
                                       predictions,
                                       attentions,
                                       p_gens,
                                       batch_oov_len,
                                       self.vocab_size,
                                       self.batch_size)

        attentions = tf.stack(attentions, axis=1)

        return tf.stack(final_dists, 1), attentions, tf.stack(coverage, 1)


def _calc_final_dist(_enc_batch_extend_vocab, vocab_dists, attn_dists, p_gens, batch_oov_len, vocab_size, batch_size):
    """
    Calculate the final distribution, for the pointer-generator model
    Args:
    vocab_dists: The vocabulary distributions. List length max_dec_steps of (batch_size, vsize) arrays.
                The words are in the order they appear in the vocabulary file.
    attn_dists: The attention distributions. List length max_dec_steps of (batch_size, attn_len) arrays
    Returns:
    final_dists: The final distributions. List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
    vocab_dists = [p_gen * dist for (p_gen, dist) in zip(p_gens, vocab_dists)]
    attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(p_gens, attn_dists)]

    # Concatenate some zeros to each vocabulary dist, to hold the probabilities for in-article OOV words
    extended_vsize = vocab_size + batch_oov_len  # the maximum (over the batch) size of the extended vocabulary
    extra_zeros = tf.zeros((batch_size, batch_oov_len))
    # list length max_dec_steps of shape (batch_size, extended_vsize)
    vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]

    # Project the values in the attention distributions onto the appropriate entries in the final distributions
    # This means that if a_i = 0.1 and the ith encoder word is w, and w has index 500 in the vocabulary,
    # then we add 0.1 onto the 500th entry of the final distribution
    # This is done for each decoder timestep.
    # This is fiddly; we use tf.scatter_nd to do the projection
    batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
    batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
    attn_len = tf.shape(_enc_batch_extend_vocab)[1]  # number of states we attend over
    batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
    indices = tf.stack((batch_nums, _enc_batch_extend_vocab), axis=2)  # shape (batch_size, enc_t, 2)
    shape = [batch_size, extended_vsize]
    # list length max_dec_steps (batch_size, extended_vsize)
    attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in attn_dists]

    # Add the vocab distributions and the copy distributions together to get the final distributions
    # final_dists is a list length max_dec_steps; each entry is a tensor shape (batch_size, extended_vsize) giving
    # the final distribution for that decoder timestep
    # Note that for decoder timesteps and examples corresponding to a [PAD] token, this is junk - ignore.
    final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                   zip(vocab_dists_extended, attn_dists_projected)]

    return final_dists