__author__ = 'fanfan'
import os
from ner.tv.gpu import ner_setting
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_word2id():
    word2id_pkl_path = ner_setting.word2id_path
    word2id_dict = None
    with open(word2id_pkl_path,'rb') as f:
        word2id_dict = pickle.load(f, encoding='iso-8859-1')

    return  word2id_dict

def read_data(path,max_length = 20,test=False):

    output_dict = dict()

    input_x = []
    input_y = []


    count = 0

    with open(path,encoding='utf-8') as f:
        for line in f:
            count += 1
            line_list = line.strip().split(' ')
            words_id = line_list[:ner_setting.max_document_length]
            tags_id = line_list[ner_setting.max_document_length:]
            words_id = [int(x) for x in words_id]
            tags_id = [int(y) for y in tags_id]

            if  words_id and len(tags_id) == len(words_id):
                if test == False:
                    input_x.append(words_id)
                    input_y.append(tags_id)
                else:
                    if np.random.randint(0, 30) == 20:
                        input_x.append(words_id)
                        input_y.append(tags_id)
            else:
                print("error:{}".format(line))

            #if count > 30000:
                #count =0
                #break



    output_dict['input_x'] = np.array(input_x)
    output_dict['input_y'] = np.array(input_y)

    #with open(bi_lstm_setting.data_dict_path,'wb') as f:
        #pickle.dump(output_dict,f)
    return output_dict


class BatchManagerTest(object):
    def __init__(self,path,batch_size):
        self.batch_size = batch_size
        data_set = read_data(path)
        self.input_x = data_set['input_x']
        self.input_y = data_set['input_y']
        self.epoch = int(len(self.input_x)/self.batch_size)



    def iterbatch(self):
        for i in range(self.epoch):
            x = self.input_x[self.batch_size * i: self.batch_size*(i+1)]
            y = self.input_y[self.batch_size * i: self.batch_size*(i+1)]
            yield x,y



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

        #X, y, test_size=0.33, random_state=42)
        self.train_input_x,self.test_input_x,self.train_input_y,self.test_input_y = \
            train_test_split(self.input_x,self.input_y,test_size=0.01,random_state=43)

        self.train_total = self.train_input_x.shape[0]
        self.train_epoch = self.train_total // self.batch_size

        self.test_total = self.test_input_x.shape[0]
        self.test_epoch = self.test_total // self.batch_size


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

    def test_iterbatch(self):
        for i in range(self.test_epoch
                       ):
            x = self.test_input_x[self.batch_size * i: self.batch_size*(i+1)]
            y = self.test_input_y[self.batch_size * i: self.batch_size*(i+1)]
            yield x,y



if __name__ == '__main__':
   manager = BatchManager(ner_setting.train_data_path, ner_setting.batch_size)
   print(manager.train_input_x.shape[0])
   print(manager.test_input_x.shape[0])