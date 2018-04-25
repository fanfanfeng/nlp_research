# create by fanfan on 2018/4/13 0013

import sys
import os
import platform
if 'window' in sys.platform:
    ProjectDir = r"E:\git-project\nlp_research\DualLSTMEncoderRankModel"
else:
    ProjectDir = r'/data/python_project/nlp_research/DualLSTMEncoderRankModel'

# 语料设置
corpus_data_path = os.path.join(ProjectDir,'data/xiaohuangji50w_nofenci.conv')
corpus_processed_path = os.path.join(ProjectDir,r'data/xiaohangji_process.txt')



filter_num = 3
max_vocab_size = 40000  #词库大小
vocabulary_path = os.path.join(ProjectDir,'data/vocab_{}.txt'.format(max_vocab_size))
embedding_size = 200
max_seq_len = 20

#转化为id形式的存储
corpus_to_id_path = os.path.join(ProjectDir,r'data/xiaohangji_process_to_ids_vocab{}.txt'.format(max_vocab_size))

# 神经网络设置参数
hiddenSize = 256
dropout = 0.5
layer_num = 2
learning_rate = 0.001
ranksize = 30

batch_size = 256
numEpochs = 10


cache_path = os.path.join(ProjectDir,'cache')
model_save_path = os.path.join(ProjectDir,'model/')
tf_log_train_path = os.path.join(ProjectDir,"train")
tf_log_valid_path = os.path.join(ProjectDir,"valid")


saveEvery = 10

