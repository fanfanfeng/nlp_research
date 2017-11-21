__author__ = 'fanfan'
from distutils.version import  LooseVersion
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import seq2seq

assert  LooseVersion(tf.__version__) >= LooseVersion('1.1')


import numpy as np
import time

# 加载数据
with open('data/letters_source.txt','r',encoding='utf-8') as f:
    source_data = f.read()

with open('data/letters_target.txt','r',encoding='utf-8') as f:
    target_data = f.read()

def extract_character_vocab(data):
    '''
    构造映射表
    '''
    special_words = ['<PAD>',"<UNK>","<GO>",'<EOS>']
    set_words = list(set([character for line in data.split("\n") for character in line]))

    int_to_vocab = {idx:word for idx,word in enumerate(special_words + set_words)}
    vocab_to_int = {word:idx for idx,word in int_to_vocab.items()}
    return int_to_vocab,vocab_to_int

source_int_to_letter,source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter,target_letter_to_int = extract_character_vocab(target_data)

# 对字母进行转换
source_int = [[source_letter_to_int.get(letter,source_letter_to_int['<UNK>']) for letter in line] for line in source_data.split('\n')]
target_int = [[target_letter_to_int.get(letter,target_letter_to_int['<UNK>']) for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split("\n")]

def get_inputs():
    '''
    模型输入tensor
    '''
    inputs = tf.placeholder(tf.int32,[None,None],name='inputs')
    targets = tf.placeholder(tf.int32,[None,None],name='targets')
    learning_rate = tf.placeholder(tf.float32,name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32,(None,),name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length,name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32,(None,),name='source_sequence_length')

    return  inputs,targets,learning_rate,target_sequence_length,max_target_sequence_length,source_sequence_length

# 在Encoder端，我们需要进行两步，第一步要对我们的输入进行Embedding，再把Embedding以后的向量传给RNN进行处理。
def get_encoder_layer(input_data,rnn_size,num_layers,source_sequence_length,source_vocab_size,encoding_embedding_size):
     '''
    构造Encoder层

    参数说明：
    - input_data: 输入tensor
    - rnn_size: rnn隐层结点数量
    - num_layers: 堆叠的rnn cell数量
    - source_sequence_length: 源数据的序列长度
    - source_vocab_size: 源数据的词典大小
    - encoding_embedding_size: embedding的大小
    '''
    # Encoder embedding
     encoder_embed_input = layers.embed_sequence(input_data,source_vocab_size,encoding_embedding_size)

     # RNN cell
     def get_lstm_cell(rnn_size):
         lstm_cell = rnn.LSTMCell(rnn_size,initializer=tf.truncated_normal_initializer)
         return lstm_cell

     cell = rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])
     encoder_output,encoder_state = tf.nn.dynamic_rnn(cell,encoder_embed_input,sequence_length=source_sequence_length,dtype=tf.float32)
     return encoder_output,encoder_state

# 对target数据进行预处理
def process_decoder_input(data,vocab_to_int):
    '''
    补充<GO>，并移除最后一个字符
    '''
    # cut掉最后一个字符
    ending = tf.strided_slice(data,[0,0],[batch_size,-1],[1,1])
    decoder_input = tf.concat([tf.fill([batch_size,1],vocab_to_int['<GO>']),ending],1)
    return decoder_input

def decoding_layer(target_letter_to_int,decoding_embedding_size,num_layers,rnn_size,target_sequence_length,max_target_sequence_length,encoder_state,decode_input):
    '''
    构造Decoder层

    参数：
    - target_letter_to_int: target数据的映射表
    - decoding_embedding_size: embed向量大小
    - num_layers: 堆叠的RNN单元数量
    - rnn_size: RNN单元的隐层结点数量
    - target_sequence_length: target数据序列长度
    - max_target_sequence_length: target数据序列最大长度
    - encoder_state: encoder端编码的状态向量
    - decoder_input: decoder端输入
    '''
    # 1. Embedding
    target_vocab_size = len(target_letter_to_int)
    decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size,decoding_embedding_size]))
    decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings,decode_input)

    # 2 构造Decoder中的RNN单元
    def get_decoder_cell(rnn_size):
        decoder_cell = rnn.LSTMCell(rnn_size,initializer=tf.random_normal_initializer)
        return decoder_cell
    cell = rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

    # 3. Output全连接层
    output_layer = Dense(target_vocab_size,kernel_initializer=tf.truncated_normal_initializer)

    # 4 Trainning decoder
    with tf.variable_scope('decoder_layer'):
        # 得到help对象
        training_helper = seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                 sequence_length=target_sequence_length,
                                                 time_major=False)
        # 构造decoder
        training_decoder = seq2seq.BasicDecoder(cell,training_helper,encoder_state,output_layer)
        (training_decoder_output,_,_) = seq2seq.dynamic_decode(training_decoder,impute_finished=True,maximum_iterations=max_target_sequence_length)

    # 5. Predicting decoder
    # 和training 共享参数
    with tf.variable_scope('decoder_layer',reuse=True):
        # 创建一个常量tensor并复制为batch_size的大小
        start_tokens = tf.tile(tf.constant([target_letter_to_int['<GO>']],dtype=tf.int32),[batch_size])
        predicting_helper = seq2seq.GreedyEmbeddingHelper(decoder_embeddings,start_tokens,target_letter_to_int['<EOS>'])
        predicting_decoder = seq2seq.BasicDecoder(cell,predicting_helper,encoder_state,output_layer)
        (predicting_decoder_output,_,_)= seq2seq.dynamic_decode(predicting_decoder,impute_finished=True,maximum_iterations=max_target_sequence_length)

    return training_decoder_output,predicting_decoder_output

def seq2seq_model(input_data,target,target_sequence_length,max_target_sequence_length,source_sequence_length,
                  source_vocab_size,encoder_embedding_size,decoder_embedding_size,rnn_size,num_layers):
    _,encoder_state = get_encoder_layer(input_data,rnn_size,num_layers,source_sequence_length,source_vocab_size,encoder_embedding_size)
    decoder_input = process_decoder_input(target,target_letter_to_int)
    training_decoder_output,predicting_decoder_output = decoding_layer(target_letter_to_int,
                                                                       decoder_embedding_size,
                                                                       num_layers,
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       encoder_state,
                                                                       decoder_input)
    return training_decoder_output,predicting_decoder_output

# 超参数
# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
    training_decoder_output,prediction_decoder_output = seq2seq_model(input_data,
                                                                      targets,
                                                                      target_sequence_length,
                                                                      max_target_sequence_length,
                                                                      source_sequence_length,
                                                                      len(source_letter_to_int),
                                                                      encoding_embedding_size,
                                                                      decoding_embedding_size,
                                                                      rnn_size,
                                                                      num_layers)
    training_logits = tf.identity(training_decoder_output.rnn_output,'logits')
    predicting_logits = tf.identity(prediction_decoder_output.sample_id,name='predictions')
    masks = tf.sequence_mask(target_sequence_length,max_target_sequence_length,dtype=tf.float32,name='masks')
    with tf.variable_scope("optimization_layer"):
        cost = seq2seq.sequence_loss(training_logits,targets,masks)
        optimizer = tf.train.AdamOptimizer(lr)

        train_vars = tf.trainable_variables()
        grads,_ = tf.clip_by_global_norm(tf.gradients(cost,train_vars),5)

        train_op = optimizer.apply_gradients(zip(grads,train_vars))


def pad_sentence_batch(sentence_batch,pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int]*(max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(targets,sources,batch_size,source_pad_int,target_pad_int):
    '''
    定义生成器，用来获取batch
    '''
    for batch_i in range(0,len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch,source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch,target_pad_int))

        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))
        yield pad_targets_batch,pad_sources_batch,targets_lengths,source_lengths

train_source = source_int[batch_size:]
train_target = target_int[batch_size:]

valid_source = source_int[:batch_size]
valid_target = target_int[:batch_size]

(valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(get_batches(valid_target, valid_source, batch_size,
                           source_letter_to_int['<PAD>'],
                           target_letter_to_int['<PAD>']))

display_step = 50
checkpoint = './trained_mode_ckpt'
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(1,epochs + 1):
        for batch_i,(targets_batch,source_batch,targets_lengths,source_lengths) in enumerate(get_batches(train_target, train_source, batch_size,
                           source_letter_to_int['<PAD>'],
                           target_letter_to_int['<PAD>'])):
            _,loss = sess.run([train_op,cost],{
                input_data:source_batch,
                targets:targets_batch,
                lr:learning_rate,
                target_sequence_length:targets_lengths,
                source_sequence_length:source_lengths
            })

            if batch_i % display_step == 0:
                validation_loss = sess.run(
                    [cost],
                    {
                        input_data:valid_sources_batch,
                        targets:valid_targets_batch,
                        lr:learning_rate,
                        target_sequence_length:valid_targets_lengths,
                        source_sequence_length:valid_sources_lengths
                    }
                )
                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(train_source) // batch_size,
                              loss,
                              validation_loss[0]))

    saver = tf.train.Saver()
    saver.save(sess,checkpoint)
    print("model Trained and saved")



# 预测
def source_to_seq(text):
    sequence_length = 7
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [source_letter_to_int['<PAD>']]*(sequence_length-len(text))

imput_word = 'common'
text = source_to_seq(imput_word)
loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(checkpoint+'.meta')
    loader.restore(sess,checkpoint)

    input_data = loaded_graph.get_tensor_by_name('inputs:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

    answer_logits = sess.run(logits,{
        input_data:[text]*batch_size,
        target_sequence_length:[len(imput_word)] * batch_size,
        source_sequence_length:[len(imput_word)] * batch_size
    })[0]

    pad = source_letter_to_int["<PAD>"]
    print('原始输入:', imput_word)
    print('\nSource')
    print('  Word 编号:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text])))

    print('\nTarget')
    print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format(" ".join([target_int_to_letter[i] for i in answer_logits if i != pad])))