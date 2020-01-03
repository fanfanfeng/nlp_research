# create by fanfan on 2019/12/10 0010
import re
import jieba
import random
from utils.text_util import is_chinese_uchar
import os

class Data_Prepare(object):

    def readfile(self, filename):
        texta = []
        textb = []
        tag = []
        with open(filename, 'r',encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                texta.append(self.pre_processing(line[0]))
                textb.append(self.pre_processing(line[1]))
                tag.append(line[2])
        # shuffle
        index = [x for x in range(len(texta))]
        random.shuffle(index)
        texta_new = [texta[x] for x in index]
        textb_new = [textb[x] for x in index]
        tag_new = [tag[x] for x in index]

        type = list(set(tag_new))
        dicts = {}
        tags_vec = []
        for x in tag_new:
            if x not in dicts.keys():
                dicts[x] = 1
            else:
                dicts[x] += 1
            temp = [0] * len(type)
            temp[int(x)] = 1
            tags_vec.append(temp)
        print(dicts)
        return texta_new, textb_new, tags_vec

    def pre_processing(self, text):
        # 删除（）里的内容
        text = re.sub('（[^（.]*）', '', text)
        # 只保留中文部分
        text = ''.join([x for x in text if is_chinese_uchar(x)])
        # 利用jieba进行分词
        words = ' '.join(jieba.cut(text)).split(" ")
        # 不分词
        words = [x for x in ''.join(words)]
        return ' '.join(words)

    def build_vocab(self, sentences, path,min_freq=1):
        if os.path.exists(path):
            return
        vocab = {}
        for sentence in sentences:
            for word in sentence.split(" "):
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

        vocab = {key: value for key, value in vocab.items() if value >= min_freq}
        _START_VOCAB = ['[PAD]', '[UNK]']
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        with open(path, 'w', encoding='utf-8') as fwrite:
            for word in vocab_list:
                fwrite.write(word + "\n")

def pad_sentence(sentence, max_sentence,vocabulary):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence
    '''
    UNK_ID = vocabulary.get('[UNK]')
    PAD_ID = vocabulary.get('[PAD]')
    sentence_batch_ids = [vocabulary.get(w, UNK_ID) for w in sentence]
    if len(sentence_batch_ids) > max_sentence:
        sentence_batch_ids = sentence_batch_ids[:max_sentence]
    else:
        sentence_batch_ids = sentence_batch_ids + [PAD_ID] * (max_sentence - len(sentence_batch_ids))

    if max(sentence_batch_ids) == 0:
        print(sentence)
    return sentence_batch_ids

def load_vocab(path):
    vocab_list = []
    with open(path, 'r', encoding='utf-8') as fread:
        for word in fread:
            vocab_list.append(word.strip())
    return {key: index for index, key in enumerate(vocab_list)}, vocab_list


if __name__ == '__main__':
    data_pre = Data_Prepare()
    data_pre.readfile('data/train.txt')