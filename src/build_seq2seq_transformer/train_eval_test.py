import os
import json
from tqdm import tqdm
from rouge import Rouge
import tensorflow as tf

from src.build_seq2seq_transformer.models.transformer import PGN_TRANSFORMER
from src.build_seq2seq_transformer.train_helper import train_model
from src.build_seq2seq_transformer.test_helper import beam_decode, greedy_decode

from src.build_seq2seq_pgn.batcher import batcher
from utils.config import  test_seg_path
from utils.w2v_helper import Vocab


def train(params):
    assert params["mode"].lower() == "train", "change training mode to 'train'"

    print("Creating the vocab ...")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count

    print("Creating the batcher ...")
    batch, params['steps_per_epoch'] = batcher(vocab, params)
    print(f'Total {params["steps_per_epoch"] * params["batch_size"]} examples')

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

    vocab = Vocab(params["vocab_path"], params["vocab_size"])
    params['vocab_size'] = vocab.count

    print("Creating the batcher ...")
    dataset, params['steps_per_epoch'] = batcher(vocab, params)

    print("Building the model ...")
    model = PGN_TRANSFORMER(params)

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
        with tqdm(total=params['steps_per_epoch'], position=0, leave=True) as pbar:
            for i in range(params['steps_per_epoch']):
                trial = next(gen)
                results.append(trial.abstract)
                pbar.update(1)

    get_rouge(results)
    return results


def get_rouge(results):
    # 读取结果
    # seg_test_report = pd.read_csv(test_seg_path, header=None).iloc[:len(results), 5].tolist()
    # seg_test_report = [' '.join(str(token) for token in str(line).split()) for line in seg_test_report]
    seg_test_report = [line.strip() for line in open(val_x_seg_path, 'r', encoding='utf8').readlines()][:len(results)]
    rouge_scores = Rouge().get_scores(results, seg_test_report, avg=True)
    print_rouge = json.dumps(rouge_scores, indent=2)
    with open(os.path.join(os.path.dirname(test_seg_path), 'results.csv'), 'w', encoding='utf8') as f:
        json.dump(list(zip(results, seg_test_report)), f, indent=2, ensure_ascii=False)
    print('*' * 8 + ' rouge score ' + '*' * 8)
    print(print_rouge)


def predict_result(params):
    # 预测结果
    test_and_save(params)


if __name__ == '__main__':
    pass
