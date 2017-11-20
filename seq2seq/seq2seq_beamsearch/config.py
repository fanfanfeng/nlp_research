# create by fanfan on 2017/8/25 0025
import tensorflow as tf
import os


projectDir =  os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(projectDir,'data')


# 训练数据相关参数
#source_vocabulary = 'data/vocab_enc.in' # source 的词库存储路径
#target_vocabulary = 'data/vocab_dec.in' # target的词库存储路径,用于训练对话，词库是同一个

# 训练数据相关参数
source_vocabulary = 'data/vocab40000.in' # source 的词库存储路径
target_vocabulary = 'data/vocab40000.in' # target的词库存储路径,用于训练对话，所以词库是同一个


# tensorflow网络设置参数
cell_type =  'lstm'             # 编码跟解码截断的RNN cell类型 , 默认: lstm
attention_type =  'bahdanau'    # 注意力机制类型: (bahdanau, luong), 默认: bahdanau
hidden_units =  256            # 隐层神经元个数
depth = 2                       # 神经网络的层数
embedding_size =  100           # 编码跟解码输出的向量Embedding dimensions
num_encoder_symbols = 40000     # source 的词库大小
num_decoder_symbols = 40000    # target 的词库大小
use_residual  = True            # 在每一层之间是不是使用残差网络
attn_input_feeding = False      # Use input feeding method in attentional decoder
use_dropout =  True             # 是否对rnn cell使用dropout
dropout_rate = 0.3              # Dropout probability for input/output/state units (0.0: no dropout)')

# 训练参数
learning_rate = 0.0002          # 学习率
max_gradient_norm =  3.0        # 梯度截断值
batch_size = 128                # Batch size
max_epochs =  200                # 训练多少轮
max_load_batches = 20           # 一次最多加载的训练数量
max_seq_length = 30             # 句子最大长度
display_freq = 50              # 训练多少次，然后显示一次训练结果
save_freq = 1150                # 训练多少次，保存一次结果
valid_freq = 1150               # 训练多少次，在验证集上面验证模型
optimizer = 'adam'              # 训练时候的优化器: (adadelta, adam, rmsprop)
model_dir = 'model/'            # 模型保存位置
model_name = 'movie_error_find.ckpt'        # 模型保存的名称
shuffle_each_epoch = True       # 每训练一轮是否打乱训练集
sort_by_length = True           # 根据句子长度排列训练集
use_fp16 =  False               # Use half precision float16 instead of float32 as dtype
beam_with = 5








