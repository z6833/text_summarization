import time
import tensorflow as tf

from src.build_seq2seq_transformer_pgn.schedules.lr_schedules import CustomSchedule
from src.build_seq2seq_transformer_pgn.layers.transformer import create_masks

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 1))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def train_model(model, dataset, params, ckpt_manager):
    learning_rate = CustomSchedule(params["d_model"])
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         beta_1=0.9,
                                         beta_2=0.98,
                                         epsilon=1e-9)

    # 该 @tf.function 将下面函数编译计算图，加快计算
    train_step_signature = [
        tf.TensorSpec(shape=(params['batch_size'], None), dtype=tf.int32),
        tf.TensorSpec(shape=(params['batch_size'], None), dtype=tf.int32),
        tf.TensorSpec(shape=(params['batch_size'], params['max_dec_len']), dtype=tf.int32),
        tf.TensorSpec(shape=(params['batch_size'], params['max_dec_len']), dtype=tf.int32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    ]

    # @tf.function(input_signature=train_step_signature)
    def train_step(enc_inp, enc_extended_inp, dec_inp, dec_tar, batch_oov_len):

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(enc_inp, dec_inp)

        with tf.GradientTape() as tape:
            outputs = model(enc_inp,
                            enc_extended_inp,
                            batch_oov_len,
                            dec_inp,
                            params['training'],
                            enc_padding_mask,
                            combined_mask,
                            dec_padding_mask)
            pred = outputs["logits"]
            loss = loss_function(dec_tar, pred)
            # loss = loss_function(dec_tar,
            #                      outputs,
            #                      padding_mask,
            #                      params["cov_loss_wt"],
            #                      params['is_coverage'])

        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        return loss

    best_loss = 20
    epochs = params['epochs']
    for epoch in range(epochs):
        t0 = time.time()
        step = 0
        total_loss = 0
        # for step, batch in enumerate(dataset.take(params['steps_per_epoch'])):
        for encoder_batch_data, decoder_batch_data in dataset:
            # print('batch is ', batch[0]["enc_input"])
            loss = train_step(encoder_batch_data["enc_input"],  # shape=(16, 200)
                              encoder_batch_data["extended_enc_input"],  # shape=(16, 200)
                              decoder_batch_data["dec_input"],  # shape=(16, 50)
                              decoder_batch_data["dec_target"],  # shape=(16, 50)
                              encoder_batch_data["max_oov_len"])

            step += 1
            total_loss += loss
            if step % 10 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, total_loss / step))
                # print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, step, loss.numpy()))

        if epoch % 1 == 0:
            if total_loss / step < best_loss:
                best_loss = total_loss / step
                ckpt_save_path = ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {} ,best loss {}'.format(epoch + 1, ckpt_save_path, best_loss))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / step))
                print('Time taken for 1 epoch {} sec\n'.format(time.time() - t0))
