__author__ = 'fanfan'
from gensim.models import Word2Vec
import os
import jieba
import logging
logging.basicConfig(format='%(asctime)s: %(levelname)s:%(message)s')
logging.root.setLevel(level=logging.INFO)
logging.info('Runing %s ' % 'vector')

process_path = 'data/processed'
jieba.load_userdict('dict_modify.txt')
class Sentence(object):
    def __init__(self,path):
        self.path = path

    def __iter__(self):
        for root,_,files in os.walk(self.path):
            for file in files:
                real_path = os.path.join(root,file)
                with open(real_path,encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        yield list(jieba.cut(line))


def train():

    model_path = "data/word2vec.model"
    sen = Sentence(process_path)
    model = Word2Vec(sen,size=256,min_count=2,iter=2)
    model.save(model_path)
    model.wv.save_word2vec_format("data/word2vec.bin",binary=True)


def make_vocab():

    pkl_path = "data/word2id.pkl"
    model_path = "data/word2vec.model"
    model = Word2Vec.load(model_path)
    list = sorted(model.wv.vocab,key=model.wv.vocab.get,reverse=True)
    print('len:'+ str(len(list)))
    print(list[3])

    word_to_id_dict = {key:value.index for key,value in model.wv.vocab.items()}
    import pickle
    with open(pkl_path,'wb') as f:
        pickle.dump(word_to_id_dict,f)



if __name__ == '__main__':
    make_vocab()

