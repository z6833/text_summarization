# coding=utf-8
import sys
import os
import tensorflow as tf
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from utils.config import (train_x_seg_path, train_y_seg_path,
                          valid_x_seg_path, valid_y_seg_path, test_x_seg_path,
                          epochs, vocab_key_to_index_path,
                          transformer_checkpoint_dir, results_dir)
from src.build_seq2seq_transformer.train_eval_test import train, test, predict_result


def main():
    parser = argparse.ArgumentParser()

    # 模型参数
    parser.add_argument("--max_enc_len", default=200, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=40, help="Decoder input max sequence length", type=int)
    parser.add_argument("--max_dec_steps", default=100, help="maximum number of words of the predicted abstract", type=int)
    parser.add_argument("--min_dec_steps", default=5, help="Minimum number of words of the predicted abstract", type=int)
    parser.add_argument("--batch_size", default=32, help="batch size", type=int)
    parser.add_argument("--buffer_size", default=10, help="buffer size", type=int)
    parser.add_argument("--beam_size", default=3, help="beam size for beam search decoding (must be equal to batch size in decode mode)", type=int)
    parser.add_argument("--vocab_size", default=10000, help="Vocabulary size", type=int)
    parser.add_argument("--embed_size", default=256, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=256, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=256, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--attn_units", default=256, help="[context vector, decoder state, decoder input] feedforward result dimension - this result is used to compute the attention weights", type=int)
    parser.add_argument("--learning_rate", default=0.001, help="Learning rate", type=float)
    parser.add_argument("--adagrad_init_acc", default=0.1, help="Adagrad optimizer initial accumulator value. Please refer to the Adagrad optimizer API documentation on tensorflow site for more details.", type=float)
    parser.add_argument("--max_grad_norm", default=0.8, help="Gradient norm above which gradients must be clipped", type=float)
    parser.add_argument('--eps', default=1e-12, type=float)
    parser.add_argument('--cov_loss_wt', default=0.5, help='Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize coverage loss.', type=float)
    parser.add_argument("--train_seg_x_dir", default=train_x_seg_path, help="train_seg_x_dir", type=str)
    parser.add_argument("--train_seg_y_dir", default=train_y_seg_path, help="train_seg_y_dir", type=str)

    parser.add_argument("--val_seg_x_dir", default=valid_x_seg_path, help="val_x_seg_path", type=str)
    parser.add_argument("--val_seg_y_dir", default=valid_y_seg_path, help="val_y_seg_path", type=str)

    parser.add_argument("--test_seg_x_dir", default=test_x_seg_path, help="train_seg_x_dir", type=str)
    parser.add_argument("--test_save_dir", default=results_dir, help="train_seg_x_dir", type=str)

    parser.add_argument("--checkpoint_dir", default=transformer_checkpoint_dir, help="checkpoint_dir", type=str)
    parser.add_argument("--transformer_model_dir", default=transformer_checkpoint_dir, help="Model folder")
    parser.add_argument("--model_path", help="Path to a specific model", default="", type=str)
    parser.add_argument("--log_file", help="File in which to redirect console outputs", default="", type=str)

    parser.add_argument("--epochs", default=epochs, help="train epochs", type=int)
    parser.add_argument("--vocab_path", default=vocab_key_to_index_path, help="vocab path", type=str)

    # others
    parser.add_argument("--checkpoints_save_steps", default=10, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--max_steps", default=10000, help="Max number of iterations", type=int)
    parser.add_argument("--num_to_test", default=20000, help="Number of examples to test", type=int)
    parser.add_argument("--max_num_to_eval", default=5, help="max_num_to_eval", type=int)

    # transformer
    parser.add_argument('--d_model', default=768, type=int, help="hidden dimension of encoder/decoder")
    parser.add_argument('--num_blocks', default=3, type=int, help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=8, type=int, help="number of attention heads")
    parser.add_argument('--dff', default=1024, type=int, help="hidden dimension of feedforward layer")
    parser.add_argument('--dropout_rate', default=0.1, type=float)

    # mode
    parser.add_argument("--mode", default='train', help="training, eval or test options")
    parser.add_argument("--model", default='PGN', help="which model to be slected")
    parser.add_argument("--pointer_gen", default=False, help="training, eval or test options")
    parser.add_argument("--is_coverage", default=True, help="is_coverage")
    parser.add_argument("--greedy_decode", default=False, help="greedy_decoder")
    parser.add_argument("--transformer", default=False, help="transformer")
    parser.add_argument("--decode_mode", default='greedy', help="transformer")

    args = parser.parse_args()
    params = vars(args)

    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')

    if params["mode"] == "train":
        params["training"] = True
        train(params)

    elif params["mode"] == "test":
        params["batch_size"] = params["beam_size"] = 32
        params["training"] = False
        params["decode_mode"] = 'greedy'
        # params["decode_mode"] = 'beam'
        params["print_info"] = True

        predict_result(params)


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    main()
