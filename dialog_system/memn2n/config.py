# create by fanfan on 2018/8/24 0024
import os
ROOT_DIR = r'E:\git-project\nlp_research\dialog_system\memn2n'

learning_rate = 0.01  # 学习率
anneal_rate =25 # 学习率减半epoch
anneal_stop_epoch = 100  # 学习率衰减总共epoch
max_grad_norm = 40 # 梯度截断阈值
evaluation_interval = 10 # 每隔多少轮，评估一次模型
batch_size = 32 # 一次训练的数目
hops = 3 # The number of hops. A hop consists of reading and addressing a memory slot.
epochs = 100 # 训练总的epoch
embedding_size = 20 # 词向量的维度
memory_size = 50 # 记忆的最大size
task_id = 1 #任务id,用的bABI的训练集，总共有20个任务，
data_dir = os.path.join(ROOT_DIR,r"data\tasks_1-20_v1-2\en") #训练数据目录


output_file = os.path.join(ROOT_DIR,r'data\scores.csv')

