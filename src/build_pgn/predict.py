# coding=utf-8
import tensorflow as tf

from utils.params_utils import get_params
from utils.log_utils import logger
from src.build_pgn.model import PGN
from utils.w2v_helper import Vocab
from utils.config import pgn_checkpoint_dir

def test(params):

    assert params['mode'].lower in ['test', 'eval'], 'change training mode to `test` or `eval`'

    if params['decode_mode'] == 'beam':
        assert params['beam_size'] == params['batch_size'], 'beam size must be equal to batch size .'

    model = PGN(params)
    logger.info('Building the model ...')

    vocab = Vocab(params['vocab_path'], params['vocab_size'])
    params['vocab_size'] = vocab.count
    logger.info('Creating the vocabulary ...')

    checkpoint = tf.train.Checkpoint(PGN=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=pgn_checkpoint_dir, max_to_keep=5)
    if checkpoint_manager.latest_checkpoint:
        checkpoint_manager.restore(checkpoint_manager.latest_checkpoint)
        logger.info(f'Building model by restoring {checkpoint_manager.latest_checkpoint}')
    else:
        logger.info('Building model from initial ...')
    logger.info('Model restored .')


if __name__ == '__main__':
    # 获得参数
    params = get_params()

    # beam search
    params['batch_size'] = 32
    params['beam_size'] = 4
    params['mode'] = 'test'
    params['decode_mode'] = 'greedy'
    params['pointer_gen'] = True
    params['use_coverage'] = True
    params['enc_units'] = 128
    params['dec_units'] = 256
    params['attn_units'] = 128
    params['min_dec_steps'] = 3
    params['max_enc_len'] = 200
    params['max_dec_len'] = 40

    # greedy search
    # params['batch_size'] = 8
    # params['mode'] = 'test'
    # params['decode_mode'] = 'greedy'
    # params['pointer_gen'] = True
    # params['use_coverage'] = False
    # params['enc_units'] = 256
    # params['dec_units'] = 512
    # params['attn_units'] = 256
    # params['min_dec_steps'] = 3

    test(params)