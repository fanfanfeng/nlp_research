# create by fanfan on 2017/7/10 0010
import os
import tensorflow as tf


source_type = 1 # 1，ner_tv的字向量
                # 2，ner_tv的词向量

if source_type == 1:
    word2vec_path = r'E:\tv_category\word_vec_mode\ner_right_train_pingshu_train\w2v_ner.pkl'
    word2id_path = r'E:\tv_category\word_vec_mode\ner_right_train_pingshu_train\word2id_ner.pkl'
    tv_data_path = r'E:\tv_category\word_vec_mode\ner_right_train_pingshu_train\data'
    # 模型保存位置
    train_model_bi_lstm = r"Model/bilstm_train_model_256_single/"
    graph_model_bi_lstm = train_model_bi_lstm

elif source_type == 2:
    word2vec_path = r'E:\tv_category\w2v.model.pkl'
    word2id_path = r'E:\tv_category\word2id.pkl'
    tv_data_path = r'E:\tv_category\normal_ner_train'
    #模型保存位置
    train_model_bi_lstm = r"Model/bilstm_train_model_256/"
    graph_model_bi_lstm = r"Model/bilstm_train_graph_256/"













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
batch_size = 300 #每次批量学习的数目
sentence_length = 20 #句子长度
initial_learning_rate = 0.01 #初始学习率
min_learning_rate = 0.0001 #最小学习率
decay_rate = 0.8 #学习衰减比例
decay_step = 800 #学习率衰减步长
max_grad_norm = 5 #最大截断值
max_document_length = 20 #句子最大长度

#训练相关参数
num_epochs = 200 #重复训练的次数
valid_num = 3000 #用于验证模型的测试集数目
show_every = 20 #没训练10次，验证模型
valid_every = 200 #每训练100次，在测试集上面验证模型
checkpoint_every = 400 #没训练200，保存模型



