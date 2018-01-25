# create by fanfan on 2017/7/10 0010
import os
import sys

if 'linux' in sys.platform:
    word2vec_path = r'/data/python_project/ner_right_train_pingshu_train/w2v_ner.pkl'
    word2id_path = r'/data/python_project/ner_right_train_pingshu_train/word2id_ner.pkl'
    tv_data_path = r'/data/python_project/ner_right_train_pingshu_train/data'
else:
    word2vec_path = r'E:\tv_category\train_and_test\word2vec_ner.pkl'
    word2id_path = r'E:\tv_category\train_and_test\word2id_ner.pkl'
    train_data_path = r'E:\tv_category\train_and_test\ner_train.txt'
    test_data_path = r'E:\tv_category\train_and_test\ner_test.txt'

# 模型保存位置
train_model_bi_lstm = r"Model/bilstm_train_model_256_single/"
graph_model_bi_lstm = train_model_bi_lstm

# Command  命令类别
# Person   人名
# Place    地名
# Language  语言
# Time      时间
# Episode   第几集，第几季
# MajorNoun 统称的名字：电影名，音乐名等
# Category 类别：动作片，爱情片等，幽默，搞笑等


tag_to_id = {"O": 0,
             "B_Command": 1, "I_Command": 2,
             "B_Person": 3, "I_Person": 4,
             "B_Place": 5, "I_Place": 6,
             "B_Language":7,"I_Language":8,
             "B_Time":9,"I_Time":10,
             "B_Episode":11,"I_Episode":12,
             'B_MajorNoun':13,"I_MajorNoun":14,
             'B_Category':15,"I_Category":16
             }

id_to_tag = { item[1]:item[0] for item in tag_to_id.items()}


#模型参数
tags_num = 17#分类的数目
embedding_dim = 200 #词向量的维度
hidden_neural_size = 256 #lstm隐层神经元数目
hidden_layer_num = 2 #lstm的层数8=
dropout = 0.5 #dropout的概率值
batch_size = 500 #每次批量学习的数目
initial_learning_rate = 0.1 #初始学习率
min_learning_rate = 0.0001 #最小学习率
decay_rate = 0.7 #学习衰减比例
decay_step = 2000 #学习率衰减步长
max_grad_norm = 5 #最大截断值
max_document_length = 20 #句子最大长度

#训练相关参数
num_epochs = 30 #重复训练的次数
step_per_epochs = 2000
show_ervery = 50 #每训练50步，打印训练的loss
checkpoint_every = 200 #没训练200，保存模型



