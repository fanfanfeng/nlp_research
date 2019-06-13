# create by fanfan on 2017/8/25 0025
import os


projectDir =  os.path.dirname(os.path.abspath(__file__))

class TestParams:
    origin_data = os.path.join(projectDir,'data')
    output_path = os.path.join(projectDir,"output")

    # tensorflow网络设置参数
    cell_type =  'gru'             # 编码跟解码截断的RNN cell类型 , 默认: lstm
    attention_type =  'bahdanau'    # 注意力机制类型: (bahdanau, luong), 默认: bahdanau
    hidden_units =  50            # 隐层神经元个数
    attention_size = 60
    depth = 4                       # 神经网络的层数
    embedding_size =  70           # 编码跟解码输出的向量Embedding dimensions
    vocab_size = 100   # 词库大小
    use_residual  = True            # 在每一层之间是不是使用残差网络
    use_dropout =  True             # 是否对rnn cell使用dropout
    dropout_rate = 0.5              # Dropout probability for input/output/state units (0.0: no dropout)')

    # 训练参数
    learning_rate = 0.0001          # 学习率
    max_gradient_norm =  3.0        # 梯度截断值
    batch_size = 2                # Batch size
    max_epochs =  10                # 训练多少轮
    max_seq_length = 20             # 句子最大长度
    display_freq = 50              # 训练多少次，然后显示一次训练结果
    valid_freq = 10               # 训练多少次，在验证集上面验证模型
    optimizer = 'adam'              # 训练时候的优化器: (adadelta, adam, rmsprop)
    filter_size = 5  # 过滤词频大小

    model_path = os.path.join(output_path,'model/chat')         # 模型保存位置
    device_map = "0"

    #decode模式下面，最大搜索深度
    beam_with = 3


class Params:
    origin_data = os.path.join(projectDir,'data')
    output_path = os.path.join(projectDir,"output")

    # tensorflow网络设置参数
    cell_type =  'gru'             # 编码跟解码截断的RNN cell类型 , 默认: lstm
    attention_type =  'bahdanau'    # 注意力机制类型: (bahdanau, luong), 默认: bahdanau
    hidden_units =  256            # 隐层神经元个数
    attention_size = 128
    depth = 4                       # 神经网络的层数
    embedding_size =  200           # 编码跟解码输出的向量Embedding dimensions
    vocab_size = 0   # 词库大小
    filter_size = 5 # 过滤词频大小
    use_residual  = True            # 在每一层之间是不是使用残差网络
    use_dropout =  True             # 是否对rnn cell使用dropout
    dropout_rate = 0.5              # Dropout probability for input/output/state units (0.0: no dropout)')

    # 训练参数
    learning_rate = 0.0001          # 学习率
    max_gradient_norm =  3.0        # 梯度截断值
    batch_size = 256                # Batch size
    max_epochs =  200                # 训练多少轮
    max_seq_length = 20             # 句子最大长度
    display_freq = 50              # 训练多少次，然后显示一次训练结果
    valid_freq = 50               # 训练多少次，在验证集上面验证模型
    optimizer = 'adam'              # 训练时候的优化器: (adadelta, adam, rmsprop)


    model_path = os.path.join(output_path,'model/chat')         # 模型保存位置
    device_map = "0"

    #decode模式下面，最大搜索深度
    beam_with = 1









