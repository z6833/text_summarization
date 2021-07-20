# coding=utf-8
import tensorflow as tf


def calc_loss(real, pred, dec_mask, attentions, cov_loss_wt, eps):

    log_loss = pgn_log_loss_function(real, pred, dec_mask, eps)

    cov_loss = _coverage_loss(attentions, dec_mask)

    return log_loss + cov_loss_wt * cov_loss, log_loss, cov_loss


def pgn_log_loss_function(real, final_dists, padding_mask, eps):

    loss_per_step = []
    batch_nums = tf.range(0, limit=real.shape[0])
    final_dists = tf.transpose(final_dists, perm=[1, 0, 2])
    for dec_step, dist in enumerate(final_dists):
        targets = real[:, dec_step]
        indices = tf.stack((batch_nums, targets), axis=1)
        gold_probs = tf.gather_nd(dist, indices)
        losses = -tf.math.log(gold_probs + eps)
        loss_per_step.append(losses)

    _loss = _mask_and_avg(loss_per_step, padding_mask)
    return _loss


def _coverage_loss(attn_dists, padding_mask):

    attn_dists = tf.transpose(attn_dists, perm=[1, 0, 2])
    coverage = tf.zeros_like(attn_dists[0])

    covlosses = []
    for a in attn_dists:

        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])
        covlosses.append(covloss)

        coverage += a
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss


def _mask_and_avg(values, padding_mask):
    padding_mask = tf.cast(padding_mask, dtype=values[0].dtype)
    dec_lens = tf.reduce_sum(padding_mask, axis=1)
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    values_per_ex = sum(values_per_step) / dec_lens

    return tf.reduce_mean(values_per_ex)