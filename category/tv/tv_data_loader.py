# create by fanfan on 2017/7/5 0005

from classfication.classfication_setting import classify_tv_setting
import numpy as np
import os
from common import data_convert
from gensim.models import Word2Vec

def load_data(max_sentence_length = None):
    read_dir_path = classify_tv_setting.tv_data_path
    label_dir_list = os.listdir(read_dir_path)
    x_raw = []
    y = []
    label2index_dict = {l.strip(): i for i,l in enumerate(classify_tv_setting.label_list)}

    for label_dir in label_dir_list:
        if label_dir == 'thu_jieba.txt':
            continue
        label_dir_path = os.path.join(read_dir_path,label_dir)
        label_index = label2index_dict[label_dir]
        label_item = np.zeros(len(classify_tv_setting.label_list),np.float32)
        label_item[label_index] = 1
        label_file_list = os.listdir(label_dir_path)
        for label_file in label_file_list:
            if label_file.endswith(".csv"):
                continue
            with open(os.path.join(label_dir_path,label_file),'rb') as reader:
                i = 0
                for line in reader:
                    i +=1
                    text = line.decode('utf-8').replace('\n','').replace('\r','').strip()
                    x_raw.append(text)
                    y.append(label_item)

        if not max_sentence_length:
            max_sentence_length = max([len(item.split(" ") for item in x_raw)])
        x = []

        model_path = classify_tv_setting.word2vec_path
        word2vec_model = Word2Vec.load(model_path)
        text_converter = data_convert.SimpleTextConverter(word2vec_model,max_sentence_length,None)

        for sentence,sentence_leng in text_converter.transform_to_ids(x_raw):
            x.append(sentence)
    return np.array(x),np.array(y)


class tv_data_loader():
    def __init__(self):
        self.sentenct_length = classify_tv_setting.sentence_length
        self.data_X ,self.data_Y = load_data(self.sentenct_length)
        self.totla_data_number = len(self.data_Y)
        shuffle_indices = np.random.permutation(np.arange(self.totla_data_number))
        x_shuffled = self.data_X[shuffle_indices]
        y_shuffled = self.data_Y[shuffle_indices]

        # 分割训练集和测试机，用于验证
        valid_sample_index = classify_tv_setting.valid_num
        self.x_valid, self.x_train = x_shuffled[:valid_sample_index], x_shuffled[valid_sample_index:]
        self.y_valid, self.y_train = y_shuffled[:valid_sample_index], y_shuffled[valid_sample_index:]

        self.train_data_num = len(self.y_train)
        self.num_batch = self.train_data_num//classify_tv_setting.batch_size

    def training_iter(self):
        train_data = np.array(list(zip(self.x_train,self.y_train)))
        shuffle_indices = np.random.permutation(np.arange(self.train_data_num))

        shuffle_data = train_data[shuffle_indices]
        for i in range(self.num_batch + 1):
            if i == self.num_batch:
                yield shuffle_data[i*classify_tv_setting.batch_size :]
            else:
                yield shuffle_data[i*classify_tv_setting.batch_size:(i+1) * classify_tv_setting.batch_size]





