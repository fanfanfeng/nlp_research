# create by fanfan on 2018/4/13 0013

import sys
import os
import platform
if 'win32' in sys.platform:
    ProjectDir = r"E:\git-project\nlp_research\chatbot-retrieval"
else:
    ProjectDir = r'/data/python_project/nlp_research/chatbot-retrieval'



filter_num = 5
max_vocab_size = 90000  #词库大小
vocabulary_path = os.path.join(ProjectDir,'data/vocab_{}.txt'.format(max_vocab_size))
embedding_size = 200
max_seq_len = 160



# 神经网络设置参数
hiddenSize = 256
dropout = 0.5
layer_num = 2
learning_rate = 0.001
ranksize = 30

batch_size = 256
numEpochs = 10


cache_path = os.path.join(ProjectDir,'cache/')
model_save_path = os.path.join(ProjectDir,'model/')
tf_log_train_path = os.path.join(ProjectDir,"train")
tf_log_valid_path = os.path.join(ProjectDir,"valid")


saveEvery = 10

