# create by fanfan on 2017/9/7 0007
import os
import sys



#词向量保存位置
#word2vec_path = os.path.join(project_setting.PROJECT_DIR, "Model/classfication/tv/w2v.model.pkl")
#word2id_path = os.path.join(project_setting.PROJECT_DIR,"Model/classfication/tv/word2id.pkl")
use_model_type = 0    # 0 词向量以及词向量训练集
                        # 1 字向量以及字向量训练集

use_net_work = 1    # 0 lstm
                    # 1 attention lstm
                    # 2 cnn
                    # 3 lstm + attention + cnn
attention_size = 300
if use_model_type == 0:
    if 'linux' not in sys.platform:
        word2vec_path = r'E:\tv_category\category\w2v_classify.pkl'
        word2id_path = r'E:\tv_category\category\word2id_classify.pkl'
        tv_data_path = r'E:\tv_category\category\segment_twotype'
    else:
        word2vec_path = r'/data/python_project/category/w2v_classify.pkl'
        word2id_path = r'/data/python_project/category/word2id_classify.pkl'
        tv_data_path = r'/data/python_project/category/segment_twotype'

    if use_net_work == 1:
        train_model_bi_lstm = r"Model/bilstm_attention_model_256/"
        graph_model_bi_lstm = r"Model/bilstm_attention_graph_256/"
    elif use_net_work == 2:
        train_model_bi_lstm = r"Model/cnn_model_256/"
        graph_model_bi_lstm = r"Model/cnn_graph_256/"
    elif use_net_work == 0:
        train_model_bi_lstm = r"Model/bilstm_train_model_256/"
        graph_model_bi_lstm = r"Model/bilstm_train_graph_256/"
    elif use_net_work == 3:
        train_model_bi_lstm = r"Model/bilstm_attention_cnn_model/"
        graph_model_bi_lstm = train_model_bi_lstm

elif use_model_type == 1:
    word2vec_path = r'E:\tv_category\java\w2v.model.pkl'
    word2id_path = r'E:\tv_category\java\word2id.pkl'
    tv_data_path = r'E:\tv_category\java\segment_train'
    if use_net_work == 1:
        attention_size = 300
        train_model_bi_lstm = r"Model/bilstm_attention_model_256_single/"
        graph_model_bi_lstm = train_model_bi_lstm
    elif use_net_work == 2:
        train_model_bi_lstm = r"Model/cnn_model_256_single/"
        graph_model_bi_lstm = train_model_bi_lstm
    elif use_net_work == 0 :
        train_model_bi_lstm = r"Model/bilstm_train_model_256_single_1/"
        graph_model_bi_lstm = train_model_bi_lstm
    elif use_net_work == 3:
        train_model_bi_lstm = r"Model/bilstm_attention_cnn_model_single/"
        graph_model_bi_lstm = train_model_bi_lstm

#标签列表
label_list = ['CHAT',"STORYTELL","CROSSTALK","ESSAY"] #['JOKE', 'CHAT', 'STOCK', 'TVINSTRUCTION', 'BAIKE', 'SETTING', 'PLAYER', 'VIDEO', 'MUSIC', 'APP', 'INSTRUCTION','WEATHER',"STORYTELL","CROSSTALK","ESSAY"]
index2label = {i:l.strip() for i,l in enumerate(label_list)}


#模型参数
sentence_classes = 4#分类的数目
embedding_dim = 200 #词向量的维度
hidden_neural_size = 256 #lstm隐层神经元数目
hidden_layer_num = 2 #lstm的层数8=
dropout = 0.5 #dropout的概率值
batch_size = 500 #每次批量学习的数目
sentence_length = 20 #句子长度
initial_learning_rate = 0.01 #初始学习率
min_learning_rate = 0.0001 #最小学习率
decay_rate = 0.5 #学习衰减比例
decay_step = 3000 #学习率衰减步长
max_grad_norm = 5 #最大截断值
max_document_length = 20 #句子最大长度

#训练相关参数
num_epochs = 200 #重复训练的次数
valid_num = 3000 #用于验证模型的测试集数目
show_every = 20 #没训练10次，验证模型
valid_every = 200 #每训练100次，在测试集上面验证模型
checkpoint_every = 620 #没训练200，保存模型








