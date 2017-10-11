# create by fanfan on 2017/7/11 0011
import tensorflow as  tf

from setting import ner_tv
import numpy as np
from gensim.models import  Word2Vec
import pickle

def make_word2id_dict_from_gensim(model_path,word2id_path):
    model = Word2Vec.load(model_path)
    word2id =  {}
    for index,word in enumerate(model.wv.index2word):
        word2id[word] = index

    with open(word2id_path,'wb') as fwrite:
        pickle.dump(word2id,fwrite)
    print(word2id)


#将gensim 的对象转化为一个 numpy array
def load_w2v(model_path):
    w2v_path = model_path
    model = Word2Vec.load(w2v_path)
    array_list = []

    for i in model.wv.index2word:
        array_list.append(model.wv[i])
    return np.array(array_list)


#加载数据，并转化为numpy array
def load_data(path):

    #获取gensim word2vec所在目录
    word2vec_path = ner_tv.word2vec_path
    word2vec = Word2Vec.load(word2vec_path)

    #句号的index,若果句子补助最大长度，用句号去补足
    endChar_id = 0
    sentence_words = []
    sentence_words_id =[]
    sentence_tags = []
    sentence_tags_id = []

    with open(path,'rb') as f:
        for index, line in enumerate(f):
            line = line.decode('utf-8').replace("\r","").replace("\n","")
            line_list = line.split(" ")

            words = []
            tags = []
            words_id = []
            tags_id = []
            for token in line_list:
                try:
                    tag_x, tag_y = token.split("/")
                except:
                    continue

                if tag_x not in word2vec.wv.vocab:
                    print("{}不在word2vec里面".format(tag_x))
                    continue
                words.append(tag_x)
                words_id.append(word2vec.wv.vocab[tag_x].index)
                tags.append(tag_y)
                tag_id = ner_tv.tag_to_id[tag_y]
                tags_id.append(tag_id)

                if len(words) == ner_tv.flags.sentence_length:
                    break
            length = len(words)
            if len(words) < ner_tv.flags.sentence_length:
                words += [0]*(ner_tv.flags.sentence_length - length)
                tags += ["O"] * (ner_tv.flags.sentence_length - length)
                words_id += [0] * (ner_tv.flags.sentence_length - length)
                tags_id += [0] * (ner_tv.flags.sentence_length - length)

            if words and words_id and len(words) == len(words_id):
                sentence_words.append(words)
                sentence_words_id.append(words_id)
                sentence_tags.append(tags)
                sentence_tags_id.append(tags_id)
            else:
                print("error:{}".format(line))





    return {"sentence_words":np.asarray(sentence_words),"sentence_words_id":np.asarray(sentence_words_id),
            "sentence_tags":np.asarray(sentence_tags),"sentence_tags_id":np.asarray(sentence_tags_id)}


#构造一个迭代对象
class BatchManager():
    def __init__(self,path,batch_size):
        data_read = load_data(path)
        self.sentence_words = data_read['sentence_words']
        self.sentence_tags = data_read['sentence_tags']
        self.sentence_words_id = data_read['sentence_words_id']
        self.sentence_tags_id = data_read['sentence_tags_id']

        self.len_data = len(self.sentence_words_id)
        self.num_batch = int(self.len_data/batch_size)
        self.batch_size = batch_size


    def shuffle(self):
        random_index = np.random.permutation(np.arange(self.len_data))
        self.sentence_words = self.sentence_words[random_index]
        self.sentence_tags = self.sentence_tags[random_index]
        self.sentence_words_id = self.sentence_words_id[random_index]
        self.sentence_tags_id = self.sentence_tags_id[random_index]

    def training_iter(self):
        self.shuffle()
        for i in range(self.num_batch):
            if i == self.num_batch:
                data = {"sentence_words":self.sentence_words[i * self.batch_size:],"sentence_tags":self.sentence_tags[i*self.batch_size:],"sentence_words_id":self.sentence_words_id[i*self.batch_size:],"sentence_tags_id":self.sentence_tags_id[i*self.batch_size:]}
            else:
                data = {"sentence_words":self.sentence_words[i * self.batch_size:(i+1) * self.batch_size],"sentence_tags":self.sentence_tags[i*self.batch_size:(i+1) * self.batch_size],"sentence_words_id":self.sentence_words_id[i*self.batch_size:(i+1) * self.batch_size],"sentence_tags_id":self.sentence_tags_id[i*self.batch_size:(i+1) * self.batch_size]}
            yield data


if __name__ == '__main__':
    make_word2id_dict_from_gensim(ner_tv.word2vec_path,ner_tv.dict_word2vec_path)
    #test = BatchManager(ner_tv.test_path,10)
    #for batch in test.training_iter():
        #print(batch['sentence_words'])
        #print(batch['sentence_tags'])
        #print(batch['sentence_words_id'])
        #print(batch['sentence_tags_id'])
        #break

