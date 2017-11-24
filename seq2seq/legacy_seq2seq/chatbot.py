__author__ = 'fanfan'
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import rnn
from seq2seq.legacy_seq2seq import word_token
import random
import random


# 参数设置
input_seq_len = 5  # 输入序列长度
output_seq_len = 5 #输出序列长度
PAD_ID = 0    #空值填充0
GO_ID = 1     #输入序列开始标记
EOS_ID = 2    #输入序列结尾标记

size = 20 #LSTM神经元size
init_learning_rate = 0.1 #初始学习率
min_freq = 2 #样本频率阀值

wordToken = word_token.WordToken()

question_path = r'./data/question'
answer_path = r'./data/answer'

# 放在全局的位置，为了动态算出num_encoder_symbols和num_decoder_symbols
max_token_id = wordToken.load_file_list([question_path, answer_path], min_freq)

num_encoder_symbols = max_token_id + 5
num_decoder_symbols = max_token_id + 5

def get_id_list_from(sentence):
    sentence_id_list = []
    seg_list = list(sentence)
    for str in seg_list:
        id = wordToken.word2id(str)
        if id:
            sentence_id_list.append(id)
    return sentence_id_list

def get_train_set():
    global num_encoder_symbols,num_decoder_symbols
    train_set = []
    with open(question_path,'r',encoding='utf-8') as question_file:
        with open(answer_path,'r',encoding='utf-8') as answer_file:
            while True:
                question = question_file.readline()
                answer = answer_file.readline()

                if question and answer:
                    question = question.strip()
                    answer = answer.strip()

                    question_id_list = get_id_list_from(question)
                    answer_id_list = get_id_list_from(answer)
                    if len(question_id_list) > 0 and len(answer_id_list) >0 :
                        answer_id_list.append(EOS_ID)
                        train_set.append([question_id_list,answer_id_list])
                else:
                    break
    return train_set

def get_samples(train_set,batch_num):
    """构造样本数据
    :return:
        encoder_inputs: [array([0, 0], dtype=int32), array([0, 0], dtype=int32), array([5, 5], dtype=int32),
                        array([7, 7], dtype=int32), array([9, 9], dtype=int32)]
        decoder_inputs: [array([1, 1], dtype=int32), array([11, 11], dtype=int32), array([13, 13], dtype=int32),
                        array([15, 15], dtype=int32), array([2, 2], dtype=int32)]
    """
    # train_set = [[[5, 7, 9], [11, 13, 15, EOS_ID]], [[7, 9, 11], [13, 15, 17, EOS_ID]], [[15, 17, 19], [21, 23, 25, EOS_ID]]]
    raw_encoder_input = []
    raw_decoder_input = []
    if batch_num >= len(train_set):
        batch_train_set = train_set
    else:
        random_start = random.randint(0,len(train_set) - batch_num)
        batch_train_set = train_set[random_start:random_start + batch_num]

    for sample in batch_train_set:
        raw_encoder_input.append( sample[0] + [PAD_ID] * (input_seq_len - len(sample[0])))
        raw_decoder_input.append([GO_ID] + sample[1] + [PAD_ID] * (output_seq_len - len(sample[1]) - 1))

    encoder_inputs = []
    decoder_inputs = []
    target_weights = []

    for length_idx in range(input_seq_len):
        encoder_inputs.append(
            np.array([encoder_input[length_idx] for encoder_input in raw_encoder_input],dtype=np.int32)
        )
    for length_idx in range(output_seq_len):
        decoder_inputs.append(
            np.array([decoder_input[length_idx] for decoder_input in raw_decoder_input],dtype=np.int32)
        )

        target_weights.append(
            np.array([ 0.0 if length_idx == output_seq_len - 1 or decoder_input[length_idx] == PAD_ID else 1.0 for decoder_input in raw_decoder_input])
        )

    return encoder_inputs,decoder_inputs,target_weights

def seq_to_encoder(input_seq):
    """从输入空格分隔的数字id串，转成预测用的encoder、decoder、target_weight等
    """
    input_seq_array = [ int(v) for v in input_seq.split()]
    encoder_input = input_seq_array + [PAD_ID] * (input_seq_len - len(input_seq_array))
    decoder_input = [GO_ID] + [PAD_ID] * (output_seq_len - 1)

    encoder_inputs = [ np.array([v],dtype=np.int32) for v in encoder_input ]
    decoder_inputs = [ np.array([v],dtype=np.int32) for v in decoder_input ]
    target_weights = [ np.array([1.0],dtype=np.float32)] * output_seq_len
    return encoder_inputs,decoder_inputs,target_weights

def get_model(fee_previous = False):
    learning_rate = tf.Variable(float(init_learning_rate),trainable=False,dtype=tf.float32)
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)

    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for i in range(input_seq_len):
        encoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name='encoder{0}'.format(i)))
    for i in range(output_seq_len + 1):
        decoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name='decoder{0}'.format(i)))
    for i in range(output_seq_len):
        target_weights.append(tf.placeholder(tf.float32,shape=[None],name='weight{0}'.format(i)))

    # decoder_inputs左移一个时序作为targets
    targets = [decoder_inputs[i + 1] for i in range(output_seq_len)]
    cell = rnn.BasicLSTMCell(size)

    outputs,_ = legacy_seq2seq.embedding_attention_seq2seq(
        encoder_inputs= encoder_inputs,
        decoder_inputs= decoder_inputs[:output_seq_len],
        cell= cell,
        num_encoder_symbols = num_encoder_symbols,
        num_decoder_symbols = num_decoder_symbols,
        embedding_size= size,
        output_projection=None,
        feed_previous=fee_previous,
        dtype=tf.float32
    )

    loss = legacy_seq2seq.sequence_loss(outputs,targets,target_weights)
    opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver(max_to_keep=1)

    return encoder_inputs,decoder_inputs,target_weights,outputs,loss,opt,saver,learning_rate_decay_op,learning_rate

def train():
    train_set = get_train_set()
    with tf.Session() as sess:
        encoder_inputs,decoder_inputs,target_weights,outputs,loss,update,saver,learning_rate_decay,learning_rate = get_model()
        sess.run(tf.global_variables_initializer())
        previous_losses = []
        for step in range(2000):
            sample_encoder_inputs,sample_decoder_inputs,sample_target_weights = get_samples(train_set,1000)
            input_feed = {}
            for l in range(input_seq_len):
                input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
            for l in range(output_seq_len):
                input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                input_feed[target_weights[l].name] = sample_target_weights[l]
            input_feed[decoder_inputs[output_seq_len].name] = np.zeros([len(sample_decoder_inputs[0])],dtype=np.int32)
            [loss_ret,_] = sess.run([loss,update],input_feed)
            if step % 10 == 0:
                print('step=', step, 'loss=', loss_ret, 'learning_rate=', learning_rate.eval())
                if len(previous_losses) > 5 and loss_ret > max(previous_losses[-5:]):
                    sess.run(learning_rate_decay)
                previous_losses.append(loss_ret)
                saver.save(sess,'./model/demo')

def predict():
    with tf.Session() as sess:
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate = get_model(fee_previous=True)
        saver.restore(sess,'./model/demo')
        sys.stdout.write(">")
        sys.stdout.flush()
        input_seq = sys.stdin.readline()
        while input_seq:
            input_seq = input_seq.strip()
            input_id_list = get_id_list_from(input_seq)
            if (len(input_id_list)):
                sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = seq_to_encoder(' '.join([str(v) for v in input_id_list]))
                input_feed = {}
                for l in range(input_seq_len):
                    input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
                for l in range(output_seq_len):
                    input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                    input_feed[target_weights[l].name] = sample_target_weights[l]
                input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2],dtype=np.int32)

                outputs_seq = sess.run(outputs,input_feed)
                # 因为输出数据每一个是num_decoder_symbols维的，因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
                outputs_seq = sess.run(outputs,input_feed)
                outputs_seq = [ int(np.argmax(logit[0],axis=0)) for logit in outputs_seq]
                if EOS_ID in outputs_seq:
                    outputs_seq = outputs_seq[:outputs_seq.index(EOS_ID)]
                outputs_seq = [wordToken.id2word(v) for v in outputs_seq]
                print(" ".join(outputs_seq))
            else:
                print("WARN：词汇不在服务区")
            sys.stdout.write("> ")
            sys.stdout.flush()
            input_seq = sys.stdin.readline()

if __name__ == "__main__":

        #train()

        predict()







