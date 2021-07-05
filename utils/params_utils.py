import argparse

from utils import config
from utils.file_utils import get_result_filename


def get_params():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='demotest', help="run mode", type=str)

    # 模型通用超参数
    parser.add_argument("--batch_size", default=config.batch_size, help="batch size", type=int)
    parser.add_argument("--epochs", default=config.epochs, help="train epochs", type=int)
    parser.add_argument("--learning_rate", default=0.01, help="Learning rate", type=float)

    parser.add_argument('--max_grad_norm', default=2.0, help='for gradient clipping', type=float)

    parser.add_argument("--adagrad_init_acc", default=0.1,
                        help="Adagrad optimizer initial accumulator value. Please refer to the Adagrad optimizer API "
                             "documentation on tensorflow site for more details.", type=float)
    parser.add_argument('--rand_unif_init_mag', default=0.02,
                        help='magnitude for lstm cells random uniform inititalization', type=float)
    parser.add_argument('--trunc_norm_init_std', default=1e-4,
                        help='std of trunc norm init, used for initializing everything else', type=float)
    parser.add_argument('--cov_loss_wt', default=1.0,
                        help='Weight of coverage loss (lambda in the paper). If zero, then no incentive to minimize '
                             'coverage loss.', type=float)

    # 模型语言模型相关参数
    parser.add_argument("--max_enc_len", default=config.max_enc_len, help="Encoder input max sequence length", type=int)
    parser.add_argument("--max_dec_len", default=config.max_dec_len, help="Decoder input max sequence length", type=int)

    parser.add_argument("--vocab_size", default=config.vocab_size, help="max vocab size , None-> Max ", type=int)
    parser.add_argument("--beam_size", default=config.beam_size,
                        help="beam size for beam search decoding (must be equal to batch size in decode mode)", type=int)
    parser.add_argument("--embed_size", default=config.embedding_dim, help="Words embeddings dimension", type=int)
    parser.add_argument("--enc_units", default=400, help="Encoder GRU cell units number", type=int)
    parser.add_argument("--dec_units", default=400, help="Decoder GRU cell units number", type=int)
    parser.add_argument("--att_units", default=400, help="[context vector, decoder state, decoder input] feedforward "
                                                          "result dimension - this result is used to compute the "
                                                          "attention weights", type=int)

    # 文件路径相关
    parser.add_argument("--vocab_path", default=config.vocab_key_to_index_path, help="vocab path", type=str)
    parser.add_argument("--train_seg_x_dir", default=config.train_x_seg_path, help="train_seg_x_dir", type=str)
    parser.add_argument("--train_seg_y_dir", default=config.train_y_seg_path, help="train_seg_y_dir", type=str)
    parser.add_argument("--test_seg_x_dir", default=config.test_x_seg_path, help="train_seg_x_dir", type=str)

    parser.add_argument("--checkpoint_dir", default=config.default_checkpoint_dir, help="checkpoint_dir", type=str)

    parser.add_argument("--checkpoints_save_steps", default=5, help="Save checkpoints every N steps", type=int)
    parser.add_argument("--min_dec_steps", default=4, help="min_dec_steps", type=int)

    parser.add_argument("--max_train_steps", default=config.sample_total // config.batch_size,
                        help="max_train_steps", type=int)
    parser.add_argument("--save_batch_train_data", default=False, help="save batch train data to pickle", type=bool)
    parser.add_argument("--load_batch_train_data", default=False, help="load batch train data from pickle", type=bool)

    parser.add_argument("--test_save_dir", default=config.test_save_dir, help="test_save_dir", type=str)
    parser.add_argument("--pointer_gen", default=False, help="pointer_gen", type=bool)
    parser.add_argument("--use_coverage", default=False, help="use_coverage", type=bool)

    parser.add_argument("--greedy_decode", default=False, help="greedy_decode", type=bool)
    parser.add_argument("--result_save_path", default=get_result_filename(config.batch_size, config.epochs, 200, 300),
                        help='result_save_path', type=str)

    args = parser.parse_args()
    params = vars(args)
    return params


def get_default_params():
    params = {"mode": 'train',
              "max_enc_len": 400,
              "max_dec_len": 32,
              "batch_size": config.batch_size,
              "epochs": 25,
              "vocab_path": config.vocab_path,
              "learning_rate": 0.15,
              "adagrad_init_acc": 0.1,
              "rand_unif_init_mag": 0.02,

              "trunc_norm_init_std": 1e-4,

              "cov_loss_wt": 1.0,

              "max_grad_norm": 2.0,
              "vocab_size": 31820,

              "beam_size": config.beams_size,
              "embed_size": 300,
              "enc_units": 128,
              "dec_units": 128,
              "att_units": 128,

              "train_seg_x_dir": config.train_x_seg_path,
              "train_seg_y_dir": config.train_y_seg_path,
              "test_seg_x_dir": config.test_x_seg_path,

              "checkpoints_save_steps": 5,
              "min_dec_steps": 4,

              "max_train_steps": config.sample_total // config.batch_size,
              "train_pickle_dir": '/opt/kaikeba/dataset/',
              "save_batch_train_data": False,
              "load_batch_train_data": False,

              "test_save_dir": config.test_save_dir,
              "pointer_gen": True,
              "use_coverage": True}
    return params
