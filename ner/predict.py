# create by fanfan on 2019/4/12 0012
# create by fanfan on 2017/10/11 0011

import numpy as np
from collections import OrderedDict
import  os
from ner.tf_models.base_ner_model import BasicNerModel
from ner.tf_models.bert_ner_model import BertNerModel

class Meta_Load():
    def __init__(self,model_dir,use_bert=False):
        pb_file_path = os.path.join(model_dir, 'ner.pb')
        if not use_bert:
            self.sess, self.input_node, self.output_node = BasicNerModel.load_model_from_pb(pb_file_path)
        else:
            self.sess,self.input_node,self.input_mask_node,self.output_node = BertNerModel.load_model_from_pb(pb_file_path)

        self.labels_list = []
        if os.path.exists(os.path.join(model_dir, 'label.txt')):
            with open(os.path.join(model_dir, 'label.txt'), 'r', encoding='utf-8') as fr:
                for line in fr:
                    self.labels_list.append(line.strip())

        self.vocabulary_list = []
        with open(os.path.join(model_dir, 'vocab.txt'), 'r', encoding='utf-8') as fr:
            for line in fr:
                self.vocabulary_list.append(line.strip())


        self.vocabulary_dict = {value:index for index,value in enumerate(self.vocabulary_list) }
        self.id_2_labels = {index:value for index,value in enumerate(self.labels_list)}



    def pad_sentence(self,sentence, max_sentence,vocabulary):
        '''
        对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

        参数：
        - sentence
        '''
        UNK_ID = vocabulary.get('<UNK>')
        PAD_ID = 0
        sentence_batch_ids = [vocabulary.get(w, UNK_ID) for w in sentence]
        if len(sentence_batch_ids) > max_sentence:
            sentence_batch_ids = sentence_batch_ids[:max_sentence]
        else:
            sentence_batch_ids = sentence_batch_ids + [PAD_ID] * (max_sentence - len(sentence_batch_ids))

        if max(sentence_batch_ids) == 0:
            print(sentence_batch_ids)
        return sentence_batch_ids

    def predict(self,text):
        words = list(text)
        tokens = np.array(self.pad_sentence(words,50,self.vocabulary_dict)).reshape((1,50))

        predict_ids = self.sess.run(self.output_node, feed_dict={self.input_node: tokens})
        predict_ids  = predict_ids[0].tolist()[:len(words)]

        print(self.result_to_json(text,[self.id_2_labels[i] for i in predict_ids]))
        for word,label in zip(words,predict_ids):
            print(word,label)


    def predict_bert(self,text):
        words = ['[CLS]'] + list(text) + ['[SEP]']
        tokens = np.array(self.pad_sentence(words,50,self.vocabulary_dict)).reshape((1,50))
        text_leng = len(words)
        tokens_mask = []
        for i in range(50):
            if i < text_leng:
                tokens_mask.append(1)
            else:
                tokens_mask.append(0)
        tokens_mask = np.array(tokens_mask).reshape((1,50))

        predict_ids = self.sess.run(self.output_node, feed_dict={self.input_node: tokens,self.input_mask_node:tokens_mask})
        predict_ids  = predict_ids[0].tolist()[1:len(words) -1]

        print(self.result_to_json(text,[self.id_2_labels[i] for i in predict_ids]))
        for word,label in zip(words[1:text_leng - 1],predict_ids):
            print(word,label)


    def result_to_json(self,string, tags):
        item = {
            "string": string,
            "entities": []
        }
        entity_name = ""
        entity_start = 0
        current_entity_type = ""
        idx = 0
        for char, tag in zip(string, tags):
            if current_entity_type != "" and tag != "O":
                new_entity_type = tag.replace("B_","").replace("I_","")
                if new_entity_type != current_entity_type:
                    item["entities"].append(
                        {"value": entity_name, "start": entity_start, "end": idx, "entity": current_entity_type})
                    entity_name = ""
                    entity_start = 0
                    current_entity_type = ""
            if tag[0] == "B":
                entity_name += char
                entity_start = idx
                current_entity_type = tag.replace("B_","")
            elif tag[0] == "I":
                entity_name += char
                current_entity_type = tag.replace("I_", "")
            else:
                if current_entity_type != "":
                    item["entities"].append(
                        {"value": entity_name, "start": entity_start, "end": idx-1, "entity": current_entity_type})
                entity_name = ""
                entity_start = 0
                current_entity_type = ""

            idx += 1

        if current_entity_type != "":
                item["entities"].append(
                    {"value": entity_name, "start": entity_start, "end": idx-1, "entity": current_entity_type})
        return item





def predict(use_bert=False):

    model_obj = Meta_Load(r'E:\git-project\nlp_research\ner\output',use_bert)
    while True:
        text = input("请输入句子\n")
        if use_bert:
            model_obj.predict_bert(text)
        else:
            model_obj.predict(text)





if __name__ == '__main__':
    predict()
    #for i in range(10):
    #    model_obj = Meta_Load()

