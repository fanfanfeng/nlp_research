__author__ = 'fanfan'
import os
import bi_lstm_setting
import pickle
import numpy as np
import tensorflow as tf

def load_word2id():
    word2id_pkl_path = bi_lstm_setting.word2id_path
    word2id_dict = None
    with open(word2id_pkl_path,'rb') as f:
        word2id_dict = pickle.load(f, encoding='iso-8859-1')

    return  word2id_dict

def read_data(path,max_length = 40):

    output_dict = dict()
    if os.path.exists(bi_lstm_setting.data_dict_path):
        print("reload from pkl")
        with open(bi_lstm_setting.data_dict_path,'rb') as f:
            output_dict = pickle.load(f)
    else:
        print("new pkl file")
        input_x = []
        input_y = []

        word2id_dict = load_word2id()

        count = 0
        for dir in os.listdir(path):
            real_path = os.path.join(path,dir)
            if not os.path.isdir(real_path):
                continue

            for file in os.listdir(real_path):
                real_file_path = os.path.join(real_path,file)
                if os.path.isdir(real_file_path):
                    continue
                with open(real_file_path,encoding='utf-8') as f:
                    for line in f:
                        count += 1
                        line = line.strip()
                        sentences = line.split("。")
                        for sentence in sentences:
                            tokens = [int(word2id_dict[token]) for token in sentence.split(" ") if token != "" and token in word2id_dict]
                            if len(tokens) > max_length:
                                tokens = tokens[:max_length]
                            else:
                                tokens = tokens + [0]*(max_length - len(tokens))

                            input_x.append(tokens)
                            input_y.append(bi_lstm_setting.label_list.index(dir))

                        #if count > 10000:
                           # break


        output_dict['input_x'] = np.array(input_x)
        one_hot_y = tf.one_hot(input_y, depth=9, on_value=None, off_value=None, axis=None, dtype=None, name=None)
        with tf.Session() as sess:

            output_dict['input_y'] = sess.run(one_hot_y)

        #with open(bi_lstm_setting.data_dict_path,'wb') as f:
            #pickle.dump(output_dict,f)
    return output_dict






class BatchManager(object):
    def __init__(self,path,batch_size):
        self.batch_size = batch_size
        data_set = read_data(path)
        self.input_x = data_set['input_x']
        self.input_y = data_set['input_y']
        total = self.input_x.shape[0]
        #乱序
        shuffle_index = np.random.permutation(np.arange(total))
        self.input_x = self.input_x[shuffle_index]
        self.input_y = self.input_y[shuffle_index]

        self.test_input_x = self.input_x[:3000]
        self.train_input_x = self.input_x[3000:]

        self.test_input_y = self.input_y[:3000]
        self.train_input_y = self.input_y[3000:]
        self.train_total = self.train_input_x.shape[0]
        self.train_epoch = self.train_total // self.batch_size


    def shuffle_train(self):
        shuffle_index = np.random.permutation(np.arange(self.train_total))
        self.train_input_x = self.train_input_x[shuffle_index]
        self.train_input_y = self.train_input_y[shuffle_index]


    def train_iterbatch(self):
        self.shuffle_train()
        for i in range(self.train_epoch):
            x = self.train_input_x[self.batch_size * i: self.batch_size*(i+1)]
            y = self.train_input_y[self.batch_size * i: self.batch_size*(i+1)]
            yield x,y



if __name__ == '__main__':
   manager = BatchManager(bi_lstm_setting.data_processed_path, bi_lstm_setting.batch_size)
   print(manager.train_input_x.shape[0])
   print(manager.test_input_x.shape[0])