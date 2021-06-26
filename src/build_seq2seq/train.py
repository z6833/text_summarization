# coding=utf-8
import tensorflow as tf
from os import path
from utils.w2v_helper import Vocab
from utils.gpu_utils import config_gpus
from utils.params_utils import get_params
from seq2seq_model import Seq2Seq
from train_helper import train_model


def train(params):
    # 1. 配置计算资源
    config_gpus()

    # 2. vocab
    vocab = Vocab(params['vocab_path'], params['vocab_size'])
    params['vocab_size'] = vocab.count

    # 3. 构造模型
    model = Seq2Seq(params, vocab)  # 确保传入到模型的参数params的所有制不会再被修改，不然会报错。

    # # 4. 模型存储
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint,
                                                    directory=path.join(params['checkpoint_dir'], 'seq2seq_model'),
                                                    max_to_keep=5)
    if params['restore']:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)                                        

    # 5. 模型训练
    train_model(model, vocab, params, checkpoint_manager)


if __name__ == '__main__':

    # 获取参数
    params = get_params()
    params['mode'] = 'train'

    # 模型训练
    train(params)