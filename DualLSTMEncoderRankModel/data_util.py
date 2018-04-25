# create by fanfan on 2018/4/13 0013
from DualLSTMEncoderRankModel import config
from DualLSTMEncoderRankModel.data_process.corpus import xiaohuangjidata
from utils import text_util
import os
import re
import jieba

_PAD = '_PAD'
_GO = '_GO'
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD,_GO,_EOS,_UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

def load_corpus():
    '''
    加载训练集
    :return: 
    '''
    # 将原始语料进行处理，变成一问一答的形式
    if not os.path.exists(config.corpus_processed_path):
        xiaohuangjidata.load_no_processed_xiaohuangji(config.corpus_data_path,config.corpus_processed_path)

    # 如果没有生成词汇库，根据config里面的设置，生成词汇库
    if not os.path.exists(config.vocabulary_path):
        create_vocabulary(config.vocabulary_path,config.corpus_processed_path)



_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
def basic_tokenizer(sentence):
    '''
    去除句子中的各种特殊符号
    :param sentence: 
    :return: 
    '''
    words = list(jieba.cut(sentence.strip()))
    return [w for w in words if re.split(_WORD_SPLIT,w) and w != ""]


def create_vocabulary(vocabulary_path,data_path,tokenizer = None):
    """
    根据训练集生成词库
      Args:
        vocabulary_path: 词汇库存储的地址
        data_path: 原始语料地址.
        tokenizer: 分词方法，没有就默认结巴
      """
    if not os.path.exists(vocabulary_path):
        print("从训练集 %s 中生成 词汇库 %s" % (data_path,vocabulary_path))
        vocab = {}
        with open(data_path,mode='r',encoding='utf-8') as f:
            count = 0
            for line in f:
                line = line.strip()
                count +=1
                if count % 10000 == 0:
                    print("处理了数据：%d" % count)
                # 删除句子的非中文，数字等的符号
                line = text_util.clean_text(line)

                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for word in tokens:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1

            vocab_list = _START_VOCAB + sorted(vocab,key=vocab.get,reverse=True)
            print("过滤器，词库大小：%d" % len(vocab_list))
            if len(vocab_list) > config.max_vocab_size:
                vocab_list = vocab_list[:config.max_vocab_size]

            with open(vocabulary_path,mode='w',encoding='utf-8') as vocab_write:
                for w in vocab_list:
                    vocab_write.write(w + "\n")

def load_vocabulary_from_file(vocabulary_path):
    """
        从文件中加载词库表
      Returns:
        返回一个dict,每个词对应的id
        以及所有词的一个list

      Raises:
        ValueError: 如果词库不存在
      """
    if os.path.exists(vocabulary_path):
        vocab_list = []
        with open(vocabulary_path,mode='r',encoding='utf-8') as f:
            vocab_list.extend(f.readlines())

            vocab_list = [line.strip() for line in vocab_list]
        vocab_dict = dict((x,y) for (y,x) in enumerate(vocab_list))
        return  vocab_dict,vocab_list
    else:
        raise ValueError("Vocabulary file %s not found",vocabulary_path)


def sentence_to_token_ids(sentence,vocab_dict,tokenizer=None):
    '''
    将一句话转化为 id形式
    :param sentence: 中文句子
    :param vocab_dict: 词袋
    :param tokenizer: 分词器
    :return: 
    '''
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocab_dict.get(w,UNK_ID) for w in words]

if __name__ == '__main__':
    load_corpus()
