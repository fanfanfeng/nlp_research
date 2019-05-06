# create by fanfan on 2019/4/12 0012
# create by fanfan on 2017/10/11 0011
from category.tv import classfication_setting
import tensorflow as tf
from category.tv import data_util
import jieba
import numpy as np
from collections import OrderedDict
import  os
from category.tf_models import classify_cnn_model

class Meta_Load():
    def __init__(self,model_dir):
        pb_file_path = os.path.join(model_dir, 'classify.pb')
        self.sess, self.input_node, self.output_node = classify_cnn_model.ClassifyCnnModel.load_model_from_pb(pb_file_path)

        self.intent_list = []
        if os.path.exists(os.path.join(model_dir, 'label.txt')):
            with open(os.path.join(model_dir, 'label.txt'), 'r', encoding='utf-8') as fr:
                for line in fr:
                    self.intent_list.append(line.strip())

        self.vocabulary_list = []
        with open(os.path.join(model_dir, 'vocab.txt'), 'r', encoding='utf-8') as fr:
            for line in fr:
                self.vocabulary_list.append(line.strip())


        self.vocabulary_dict = {value:index for index,value in enumerate(self.vocabulary_list) }
        self.id_2_itent = {index:value for index,value in enumerate(self.intent_list)}



    def pad_sentence(self,sentence, max_sentence,vocabulary):
        '''
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

        参数：
        - sentence
        '''
        UNK_ID = vocabulary.get('<UNK>')
        PAD_ID = vocabulary.get('_PAD')
        sentence_batch_ids = [vocabulary.get(w, UNK_ID) for w in sentence]
        if len(sentence_batch_ids) > max_sentence:
            sentence_batch_ids = sentence_batch_ids[:max_sentence]
        else:
            sentence_batch_ids = sentence_batch_ids + [PAD_ID] * (max_sentence - len(sentence_batch_ids))

        if max(sentence_batch_ids) == 0:
            print(sentence_batch_ids)
        return sentence_batch_ids

    def predict(self,text):
        jieba.load_userdict(r"E:\nlp-data\jieba_dict\dict_modify.txt")
        words = list(jieba.cut(text))
        tokens = np.array(self.pad_sentence(words,50,self.vocabulary_dict)).reshape((1,50))

        intent_pre = self.sess.run(self.output_node, feed_dict={self.input_node: tokens})
        result = { self.id_2_itent[i]: value for i, value in enumerate(intent_pre[0])}
        order_dict = OrderedDict(sorted(result.items(), key=lambda t: t[1], reverse=True))
        print(order_dict.items())






def predict():

    model_obj = Meta_Load(r'E:\git-project\nlp_research\category\output')
    while True:
        text = input("请输入句子\n")
        model_obj.predict(text)





if __name__ == '__main__':
    predict()
    #for i in range(10):
    #    model_obj = Meta_Load()

