# coding=utf-8
import tensorflow as tf
import numpy as np

from utils.params_utils import get_params
from utils.gpu_utils import config_gpus
from utils.w2v_helper import Vocab
from utils.log_utils import logger
from src.build_pgn.model import PGN
from src.build_pgn.train_helper import train_model
from src.build_pgn.batcher import batcher


def train(params):

    # GPU资源配置
    # config_gpus()

    # 配置vocab
    vocab = Vocab(params['vocab_path'], params['vocab_size'])
    params['vocab_size'] = vocab.count
    logger.info('Building the vocab object...')

    # 构建模型
    model = PGN(params)
    logger.info(f'Building the model object...')

    # 构造数据
    train_dataset, params['train_steps_per_epoch'] = batcher(vocab, params)
    valid_dataset, params['valid_steps_per_epoch'] = batcher(vocab, params)
    logger.info(f'Building the dataset for train/valid ...')

    # 构造模型保存管理器
    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, params['checkpoint_dir'], max_to_keep=5)

    if checkpoint_manager.latest_checkpoint:
        checkpoint_manager.restore(checkpoint_manager.latest_checkpoint)
        params['trained_epoch'] = int(checkpoint_manager.latest_checkpoint[-1])
        logger.info(f'Building model by restoring {checkpoint_manager.latest_checkpoint}')
    else:
        params['trained_epoch'] = 1
        logger.info('Building model from initial ...')

    # 设置学习率
    params['learning_rate'] *= np.power(0.95, params['trained_epoch'])
    logger.info(f'Learning rate : {params["learning_rate"]}')

    # 模型训练
    logger.info('Start  training model ...')
    train_model(model, train_dataset, valid_dataset, params, checkpoint_manager)


if __name__ == '__main__':
    # 获取参数
    params = get_params()
    params['mode'] = 'train'

    params['pointer_gen'] = True
    params['use_coverage'] = True

    params['enc_units'] = 128
    params['dec_units'] = 256
    params['attn_units'] = 128

    params['max_enc_len'] = 200
    params['max_dec_len'] = 40

    params['batch_size'] = 16
    params['eps'] = 1e-12

    # 模型训练
    train(params)