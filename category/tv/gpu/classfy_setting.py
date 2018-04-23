# create by fanfan on 2018/1/23 0023
import sys
use_net_work = 1    # 0 lstm
                    # 1 attention lstm
                    # 2 cnn
                    # 3 lstm + attention + cnn

if 'linux' not in sys.platform:
    word2vec_path = r'E:\tv_category\output\output\classify\word2vec.pkl'
    word2id_path = r'E:\tv_category\output\output\classify\word2id_classify.pkl'
    train_data_path = r'E:\tv_category\output\output\classify\classify_train.txt'
    test_data_path = r'E:\tv_category\output\output\classify\classify_test.txt'
else:
    word2vec_path = r'/data/python_project/output/classify/word2vec.pkl'
    word2id_path = r'/data/python_project/output/classify/word2id.pkl'
    train_data_path = r'/data/python_project/output/classify/classify_train.txt'
    test_data_path = r'/data/python_project/output/classify/classify_test.txt'



if use_net_work == 1:
    train_model_bi_lstm = r"Model/bilstm_attention_model_256/"
elif use_net_work == 2:
    train_model_bi_lstm = r"Model/cnn_model_256/"
elif use_net_work == 0:
    train_model_bi_lstm = r"Model/bilstm_train_model_256/"
elif use_net_work == 3:
    train_model_bi_lstm = r"Model/bilstm_attention_cnn_model/"
graph_model_bi_lstm = train_model_bi_lstm
attention_size = 200

#标签列表

label_list = ['CHAT',"CROSSTALK","STORYTELL","ESSAY","weather",'traintickets','live_channel_epg','radio','mv'] #['JOKE', 'CHAT', 'STOCK', 'TVINSTRUCTION', 'BAIKE', 'SETTING', 'PLAYER', 'VIDEO', 'MUSIC', 'APP', 'INSTRUCTION','WEATHER',"STORYTELL","CROSSTALK","ESSAY"]
index2label = {i:l.strip() for i,l in enumerate(label_list)}

#模型参数
sentence_classes = len(label_list)#分类的数目
embedding_dim = 200 #词向量的维度
hidden_neural_size = 256 #lstm隐层神经元数目
hidden_layer_num = 2 #lstm的层数8=
dropout = 0.5 #dropout的概率值
batch_size = 500 #每次批量学习的数目
sentence_length = 20 #句子长度
initial_learning_rate = 0.01 #初始学习率
min_learning_rate = 0.0001 #最小学习率
decay_rate = 0.7 #学习衰减比例
decay_step = 3000 #学习率衰减步长
max_grad_norm = 5 #最大截断值
max_document_length = 20 #句子最大长度

#训练相关参数
num_epochs = 15  #重复训练的次数
show_every = 30 #没训练50次，验证模型
valid_every = 400 #每训练100次，在测试集上面验证模型
total_epochs_per = 3000 # 一轮训练数
