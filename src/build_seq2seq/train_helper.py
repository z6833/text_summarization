# coding=utf-8
import time
import tensorflow as tf
from tensorflow import keras
from functools import partial

from batch_generator import train_batch_generator


# 损失函数
def loss_function(real, pred, pad_index):

    loss_obj = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.math.logical_not(tf.math.equal(real, pad_index))

    loss_ = loss_obj(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)

    loss_ *= mask
    return tf.reduce_mean(loss_)


# 批次训练
def train_step(model, enc_inputs, dec_target, initial_enc_hidden, loss_function=None, optimizer=None, mode='train'):

    with tf.GradientTape() as tape:

        # encoder部分
        enc_output, enc_hidden = model.encoder(enc_inputs, initial_enc_hidden)

        # decoder部分
        initial_dec_hidden = enc_hidden  # 用encoder的最终输出，作为第一个S_0

        # 逐个预测序列
        prediction, _ = model.teacher_decoder(initial_dec_hidden, enc_output, dec_target)

        # 预测损失
        batch_loss = loss_function(dec_target[:, 1:], prediction)

        if mode == 'train':
            variables = (model.encoder.trainable_variables + model.decoder.trainable_variables + model.attention.trainable_variables)
            gradients = tape.gradient(batch_loss, variables)
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5.)

            optimizer.apply_gradients(zip(gradients, variables))

        return batch_loss


# 模型评估
def evaluate_model(model, valid_dataset, valid_steps_per_epoch, pad_index):

    print('starting evaluating ...')

    total_loss = 0
    initial_enc_hidden = model.encoder.initialize_hidden_state()
    for batch, data in enumerate(valid_dataset.take(valid_steps_per_epoch), start=1):

        inputs, target = data

        batch_loss = train_step(model,
                                inputs,
                                target,
                                initial_enc_hidden,
                                loss_function=partial(loss_function, pad_index=pad_index),
                                mode='eval')
        total_loss += batch_loss

    return total_loss / valid_steps_per_epoch


def train_model(model, vocab, params, checkpoint_manager):

    epochs = params['epochs']

    pad_index = vocab.word2index[vocab.PAD_TOKEN]

    optimizer = keras.optimizers.Adam(name='Adam', learning_rate=params['learning_rate'])

    train_dataset, valid_dataset, train_steps_per_epoch, valid_steps_per_epoch = train_batch_generator(params['batch_size'], params['max_enc_len'], params['max_dec_len'], sample_sum=2 ** 7)

    for epoch in range(epochs):
        start_time = time.time()

        # 第一个隐状态h_0
        initial_enc_hidden = model.encoder.initialize_hidden_state()

        total_loss = 0.
        running_loss = 0.
        # 模型训练
        for batch_index, (inputs, target) in enumerate(train_dataset.take(train_steps_per_epoch), start=1):

            batch_loss = train_step(model,
                                    inputs,
                                    target,
                                    initial_enc_hidden,
                                    loss_function=partial(loss_function, pad_index=pad_index),
                                    optimizer=optimizer)

            total_loss += batch_loss

            if batch_index % 5 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch_index,
                                                             (total_loss - running_loss) / 5))
                running_loss = total_loss

        # 模型保存
        if (epoch + 1) % 1 == 0:
            ckpt_save_path = checkpoint_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        # 模型验证
        valid_loss = evaluate_model(model, valid_dataset, valid_steps_per_epoch, pad_index)

        print('Epoch {} Loss {:.4f}; val Loss {:.4f}'.format(epoch + 1,
                                                             total_loss / train_steps_per_epoch,
                                                             valid_loss))

        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start_time))

