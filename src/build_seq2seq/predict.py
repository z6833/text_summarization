# coding=utf-8

# coding=utf-8
import tensorflow as tf
from os import path
import pandas as pd
from rouge import Rouge
import json

from utils.w2v_helper import Vocab
from utils.gpu_utils import config_gpus
from utils.params_utils import get_params
from seq2seq_model import Seq2Seq
from utils.data_loader import load_dataset
from utils.config import test_x_path, test_y_path, test_seg_path
from predict_helper import greedy_decode,  beam_decode
from batch_generator import beam_test_batch_generator


def get_rouge(results):

    # 读取结果
    seg_test_report = pd.read_csv(test_seg_path, header=None).iloc[:, 5].tolis()
    seg_test_report = [' '.join(str(token) for token in line.split()) for line in seg_test_report]

    rouge_score = Rouge().get_scores(results, seg_test_report, avg=True)
    print_rouge = json.dumps(rouge_score, indent=4)

    with open(path.join(path.dirname(test_seg_path), 'results.csv'), 'w', encoding='utf-8') as f:
        json.dump(list(zip(results, seg_test_report)), f, indent=4, ensure_ascii=False)

    print('==' * 8 + print_rouge + '==' * 8)
    print(print_rouge)


def predict_result(model, params, vocab):

    test_x, _ = load_dataset(test_x_path, test_y_path, params['max_enc_len'], params['max_dec_len'])

    # 预测结果
    results = greedy_decode(model, test_x, params['batch_size'], vocab, params)
    # 结果保存
    get_rouge(results)


def test(params):

    assert params['mode'].lower() in ['test', 'eval'], 'change training mode to `test` or `eval`'
    assert params['beam_size'] == params['batch_size'], 'beam size must be equal to batch size'

    # 配置计算资源
    config_gpus()

    print('createing the vocab ...')
    vocab = Vocab(params['vocab_path'], params['vocab_size'])
    params['vocab_size'] = vocab.count

    print('building the model ...')
    model = Seq2Seq(params, vocab)

    print('building the checkpoint manager ...')
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    directory=path.join(params['checkpoint_dir'], 'seq2seq_model'),
                                                    max_to_keep=5)

    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print(f'restored model from {checkpoint_manager.latest_checkpoint}')
    else:
        print('initializing from scratch')

    if params['greedy_decode']:
        print('using greedy search to decoding .')
        predict_result(model, params, vocab)
    else:
        print('using beam search decoding ...')
        batch_data_generator = beam_test_batch_generator(params['beam_size'])
        results = []
        for batch_data in batch_data_generator:
            best_hyp = beam_decode(model, batch_data, vocab, params)
            results.append(best_hyp)
        get_rouge(results)
        print(f'save result to {params["result_save_path"]} .')


if __name__ == '__main__':

    # 获取参数
    params = get_params()
    params['mode'] = 'test'
    params['restore'] = True  # 基于原来的模型继续训练

    params['greedy_decode'] = True

    # 模型训练
    test(params)