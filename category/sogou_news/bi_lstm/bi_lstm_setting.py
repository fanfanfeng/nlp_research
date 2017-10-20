__author__ = 'fanfan'
import os
currentdir = os.path.dirname(os.path.abspath(__file__))

sentence_classes    = 9     # 分类的数目
label_list = ['IT', '体育', '健康', '军事', '招聘', '教育', '文化', '旅游', '财经']
data_label_dict = {
    'C000008': '财经',
    'C000010': 'IT',
    'C000013': '健康',
    'C000014': '体育',
    'C000016': '旅游',
    'C000020': '教育',
    'C000022': '招聘',
    'C000023': '文化',
    'C000024': '军事'
}


# rnn 模型参数设置
embedding_dim       = 256   # 词向量的维度
hidden_neural_size  = 256   # rnn隐层神经元数目
hidden_layer_num    = 3     # rnn的层数
dropout             = 0.5   # dropout的概率值

#训练参数
batch_size          = 200   # 每次批量学习的数目
sentence_length     = 40    # 句子长度
initial_learning_rate=0.01  # 初始学习率
min_learning_rate   = 0.0001# 最小学习率
decay_rate          = 0.3   # 学习衰减比例
decay_step          = 1000  # 学习率衰减步长
max_grad_norm       = 5     # 最大截断值
num_epochs          = 200   # 重复训练的次数
valid_num           = 2000  # 用于验证模型的测试集数目
show_every          = 50   # 每训练100次，输入loss值
valid_every         = 210   # 每训练200次，在测试集上面验证模型
checkpoint_every    = 400   # 每训练200次，保存模型
model_save_path     = os.path.join(currentdir,"../model/bilstm.ckpt")







#用gensim 训练的词向量位置
word2vec_path       = os.path.join(currentdir,"../data/word2vec.model")
word2id_path        = os.path.join(currentdir,"../data/word2id.pkl")
data_dict_path      = os.path.join(currentdir,"../data/data.pkl")
data_processed_path = os.path.join(currentdir,"../data/processed")
graph_save_path     = os.path.join(currentdir,"../data/graph")
#






