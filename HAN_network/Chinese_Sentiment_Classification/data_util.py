# create by fanfan on 2017/10/17 0017
# create by fanfan on 2017/10/17 0017
import numpy as np
import tensorflow as tf
import pickle
import re
from collections import Counter
import itertools
from HAN_network.Chinese_Sentiment_Classification import settings
import os
from  tqdm import tqdm

PAD = '_PAD'
UNK = '_UNK'

def Q2B(uchar):
    '''全角转半角'''
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0

    # 转完之后不是半角字符返回原来的字符
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    return chr(inside_code)

def replace_all(repls,text):
    return re.sub('|'.join(re.escape(key) for key in repls.keys()),lambda k:repls[k.group(0)],text)

def split_sentence(txt):
    sents = re.split(r'\n|\s|;|；|。|，|\.|,|\?|\!|｜|[=]{2,}|[.]{3,}|[─]{2,}|[\-]{2,}|~|、|╱|∥', txt)
    sents = [c for s in sents for c in re.split(r'([^%]+[\d,.]+%)', s)]
    sents = list(filter(None,sents))
    return  sents

def normalize_punctuation(text):
    cpun = [['	'],
            ['﹗', '！'],
            ['“', '゛', '〃', '′', '＂'],
            ['”'],
            ['´', '‘', '’'],
            ['；', '﹔'],
            ['《', '〈', '＜'],
            ['》', '〉', '＞'],
            ['﹑'],
            ['【', '『', '〔', '﹝', '｢', '﹁'],
            ['】', '』', '〕', '﹞', '｣', '﹂'],
            ['（', '「'],
            ['）', '」'],
            ['﹖', '？'],
            ['︰', '﹕', '：'],
            ['・', '．', '·', '‧', '°'],
            ['●', '○', '▲', '◎', '◇', '■', '□', '※', '◆'],
            ['〜', '～', '∼'],
            ['︱', '│', '┼'],
            ['╱'],
            ['╲'],
            ['—', 'ー', '―', '‐', '−', '─', '﹣', '–', 'ㄧ', '－']]
    epun = [' ', '!', '"', '"', '\'', ';', '<', '>', '、', '[', ']', '(', ')', '?', ':', '･', '•', '~', '|', '/', '\\', '-']
    repls = {}
    for i in range(len(cpun)):
        for j in range(len(cpun[i])):
            repls[cpun[i][j]] = epun[i]
    return  replace_all(repls,text)

def clean_str(txt):
    txt = txt.replace("  ","")
    txt = normalize_punctuation(txt)
    txt = ''.join([Q2B(c) for c in list(txt)])
    return txt

def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary = {x:i for i,x in enumerate(vocabulary_inv)}
    return [vocabulary,vocabulary_inv]

#从pkl文件里面获取vocab
def get_vocab(path = settings.vocab_pkl):
    if not os.path.exists(path) or os.path.isdir(path):
        raise ValueError("No file at {}".format(path))

    char_list = pickle.load(open(path,'rb'))
    vocab = dict(zip(char_list,range(len(char_list))))
    return vocab,char_list

#随机采样，以免训练数据不足
def upsampling(x,size):
    if len(x) > size:
        return x
    diff_size = size - len(x)
    return x + list(np.random.choice(x,diff_size,replace=False))

def write_data(doc,label,out_f,vocab):
    doc = split_sentence(clean_str(doc))
    document_length = len(doc)
    sentence_lengths = np.zeros((settings.max_doc_len,),dtype= np.int64)
    data = np.ones((settings.max_doc_len * settings.max_sentence_len,),dtype=np.int64)
    doc_len = min(document_length,settings.max_doc_len)

    for j in range(doc_len):
        sent = doc[j]
        actual_len = len(sent)
        pos = j * settings.max_sentence_len
        sent_len = min(actual_len,settings.max_sentence_len)
        sentence_lengths[j] = sent_len

        data[pos:pos+sent_len] = [vocab.get(sent[k],0) for k in range(sent_len)]

    features = {'sentence_lengths':tf.train.Feature(int64_list=tf.train.Int64List(value=sentence_lengths)),
                'document_lengths':tf.train.Feature(int64_list=tf.train.Int64List(value=[doc_len])),
                'label':tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'text':tf.train.Feature(int64_list=tf.train.Int64List(value=data))
                }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    out_f.write(example.SerializeToString())

def build_dataset(pos_path = settings.pos_data_path,
                  neg_path = settings.neg_data_path):
    pos_docs = list(open(pos_path,encoding='utf-8').readlines())
    neg_docs = list(open(neg_path,encoding='utf-8').readlines())

    #加载词库
    vocab,_ = get_vocab(settings.vocab_pkl)

    pos_size = len(pos_docs)
    neg_size = len(neg_docs)

    pos_train_size = int(pos_size * 0.9)
    pos_valid_size = pos_size - pos_train_size

    neg_train_size = int(neg_size * 0.9)
    neg_valid_size = neg_size - neg_train_size

    train_path = os.path.join(settings.data_dir,'train.tfrecords')
    valid_path = os.path.join(settings.data_dir,'valid.tfrecords')

    with tf.python_io.TFRecordWriter(train_path) as out_f:
        train_size = max(pos_train_size,neg_train_size)
        pos_train_docs = np.random.choice(upsampling(pos_docs[:pos_train_size],train_size),train_size,replace=False)
        neg_train_docs = np.random.choice(upsampling(neg_docs[:neg_train_size],train_size),train_size,replace=False)

        print(len(pos_train_docs),len(neg_train_docs))
        for i in tqdm(range(train_size)):
            pos_row = pos_train_docs[i]
            neg_row = neg_train_docs[i]
            write_data(pos_row,1,out_f,vocab)
            write_data(neg_row,0,out_f,vocab)

    with tf.python_io.TFRecordWriter(valid_path) as out_f:
        valid_size = max(pos_valid_size, neg_valid_size)
        pos_valid_docs = np.random.choice(upsampling(pos_docs[pos_train_size:], valid_size), valid_size, replace=False)
        neg_valid_docs = np.random.choice(upsampling(neg_docs[neg_train_size:], valid_size), valid_size, replace=False)

        print(len(pos_valid_docs), len(neg_valid_docs))
        for i in tqdm(range(valid_size)):
            pos_row = neg_valid_docs[i]
            neg_row = neg_train_docs[i]
            write_data(pos_row, 1, out_f,vocab)
            write_data(neg_row, 0, out_f,vocab)

    print('Done {} records, train {}, valid {}'.format(pos_size + neg_size,
                                                       pos_train_size + neg_train_size,
                                                       pos_valid_size + neg_valid_size))


class sentence_reader():
    def __init__(self):
        self.pos_docs = settings.pos_data_path
        self.neg_docs = settings.neg_data_path

    def __iter__(self):
        for line in open(self.pos_docs,encoding='utf-8'):
            line  = line.strip()
            line = line.replace("  ", "")
            line = normalize_punctuation(line)
            yield list(line)

        for line in open(self.neg_docs,encoding='utf-8'):
            line  = line.strip()
            line = line.replace("  ", "")
            line = normalize_punctuation(line)
            yield list(line)



if __name__ == '__main__':
    status = 0
    if status == 0:
        build_dataset()
