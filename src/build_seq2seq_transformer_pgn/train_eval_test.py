# coding=utf-8
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from src.build_seq2seq_transformer_pgn.models.transformer import PGN_TRANSFORMER
from src.build_seq2seq_transformer_pgn.train_helper import train_model
from src.build_seq2seq_transformer_pgn.test_helper import beam_decode, greedy_decode

from utils.w2v_helper import Vocab
from utils.config import test_data_path
from src.build_seq2seq_pgn.batcher import batcher
from utils.file_utils import get_result_filename


def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count

    print("Creating the batcher ...")
    batch, params['steps_per_epoch'] = batcher(vocab, params)

    print("Building the model ...")
    model = PGN_TRANSFORMER(params)

    print("Creating the checkpoint manager")

    checkpoint = tf.train.Checkpoint(PGN_TRANSFORMER=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, params['transformer_model_dir'], max_to_keep=5)

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
        params["trained_epoch"] = int(checkpoint_manager.latest_checkpoint[-1])
    else:
        print("Initializing from scratch.")
        params["trained_epoch"] = 1

    print("Starting the training ...")
    train_model(model, batch, params, checkpoint_manager)


def test(params):
    assert params["mode"].lower() == "test", "change training mode to 'test' or 'eval'"
    # assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch_size, change the params"

    print("Building the model ...")
    model = PGN_TRANSFORMER(params)

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating the batcher ...")
    dataset, params['steps_per_epoch'] = batcher(vocab, params)

    print("Creating the checkpoint manager")
    ckpt = tf.train.Checkpoint(PGN_TRANSFORMER=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, params['transformer_model_dir'], max_to_keep=5)

    ckpt.restore(ckpt_manager.latest_checkpoint)
    print("Model restored")

    for batch in dataset:
        if params['decode_mode'] == "greedy":
            yield greedy_decode(model, dataset, vocab, params)
        else:
            yield beam_decode(model, batch, vocab, params, params['print_info'])


def test_and_save(params):

    assert params["test_save_dir"], "provide a dir where to save the results"

    gen = test(params)
    if params['decode_mode'] == "greedy":
        results = next(gen)
    else:
        results = []
        with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
            for i in range(params["num_to_test"]):
                trial = next(gen)
                results.append(trial.abstract)
                pbar.update(1)

    return results


def predict_result(params):

    # 预测结果
    results = test_and_save(params)
    # 保存结果
    save_predict_result(results, params)


def save_predict_result(results, params):

    # 读取结果
    test_df = pd.read_csv(test_data_path)
    # 填充结果
    test_df['Prediction'] = results[:20000]
    # 　提取ID和预测结果两列
    test_df = test_df[['QID', 'Prediction']]
    # 保存结果.
    result_save_path = get_result_filename(params)
    test_df.to_csv(result_save_path, index=None, sep=',')


if __name__ == '__main__':
    pass