# coding=utf-8
import pathlib
from os import path

# 项目根目录
root_dir = pathlib.Path(path.abspath(__file__)).parent.parent
# 其他常用路径
data_dir = path.join(root_dir, 'data')
model_dir = path.join(root_dir, 'model')


# 数据集路径
train_data_path = path.join(data_dir, 'train.csv')
test_data_path = path.join(data_dir, 'test.csv')

# 预处理后的数据集路径
train_seg_path = path.join(data_dir, 'train_seg_data.csv')
test_seg_path = path.join(data_dir, 'test_seg_data.csv')

# 合并train/test，构造用于训练词向量的数据 路径
merged_seg_path = path.join(data_dir, 'merged_train_test_seg_data.csv')

# 词向量维度
embedding_dim = 100
word2vec_train_epochs = 5

# 词向量模型保存路径
save_w2v_model_path = path.join(model_dir, 'word2vector', 'word2vec.model')
# 词向量矩阵保存路径
embedding_matrix_path = path.join(model_dir, 'word2vector', 'embedding_matrix')

# 词向量词汇表保存路径
vocab_index_to_key_path = path.join(model_dir, 'word2vector', 'vocab_index_to_key.json')
vocab_key_to_index_path = path.join(model_dir, 'word2vector', 'vocab_key_to_index.json')


# 停用词路径
stopwords_path = path.join(root_dir, 'stopwords', 'stopwords.txt')

# train/test数据与标签的路径
train_x_seg_path = path.join(data_dir, 'train_x_seg_data.csv')
train_y_seg_path = path.join(data_dir, 'train_y_seg_data.csv')

test_x_seg_path = path.join(data_dir, 'test_x_seg_data.csv')
test_y_seg_path = path.join(data_dir, 'test_y_seg_data.csv')

# train/test数据与标签，pad处理后的路径
train_x_pad_path = path.join(data_dir, 'train_x_pad_data.csv')
train_y_pad_path = path.join(data_dir, 'train_y_pad_data.csv')

test_x_pad_path = path.join(data_dir, 'test_x_pad_data.csv')
test_y_pad_path = path.join(data_dir, 'test_y_pad_data.csv')

# train/set数据与标签，转换成索引形式后路径
train_x_path = path.join(data_dir, 'train_X')
train_y_path = path.join(data_dir, 'train_Y')

test_x_path = path.join(data_dir, 'test_X')
test_y_path = path.join(data_dir, 'test_Y')
