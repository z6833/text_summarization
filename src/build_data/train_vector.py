# coding=utf-8
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from utils import config
from tools import cpu_cores

def train_word2vec(file_path=config.merged_seg_path):

    # 训练词向量
    model = Word2Vec(
        LineSentence(source=file_path),
        vector_size=config.embedding_dim,
        sg=1,
        workers=cpu_cores,
        window=5,
        min_count=5,
        epochs=config.word2vec_train_epochs,
    )

    return model