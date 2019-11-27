# create by fanfan on 2018/4/13 0013

import sys
import os
if 'win32' in sys.platform:
    ProjectDir = r"E:\git-project\nlp_research\chatbot_retrieval"
else:
    ProjectDir = r'/data/python_project/nlp_research/chatbot_retrieval'


# 语料处理设置参数
min_word_frequency = 5
max_seq_len = 160

DATA_PATH = os.path.join(ProjectDir,"data")
vocabulary_path = os.path.join(DATA_PATH,'vocab_processor.txt')
vocabulary_path_bin = os.path.join(DATA_PATH,"vocab_processor.bin")
TRAIN_PATH = os.path.join(DATA_PATH, "train.csv")
VALIDATION_PATH = os.path.join(DATA_PATH, "valid.csv")
TEST_PATH = os.path.join(DATA_PATH, "test.csv")




# 神经网络设置参数
embedding_size = 200
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

