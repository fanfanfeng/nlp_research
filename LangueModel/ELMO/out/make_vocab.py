# create by fanfan on 2018/11/23 0023

from LangueModel.ELMO.out import config
import jieba
from utils import text_util
import os



def chinese_tokenizer(document):
    """
    把中文文本转为词序列
    """
    tokens = list(jieba.cut(document))
    tokens = [word for word in tokens if text_util.is_chinese_uchar(word)]
    # 分词

    return tokens


def create_vocabulary(vocabulary_path,data_path,tokenizer = None,add_tokens=[],filter_num = 0):
    '''
    根据data_path创建词库表
    :param vocabulary_path: 词库表存储位置
    :param data_path: 原始数据位置
    :param tokenizer: 分词器
    :return: 
    '''
    if not os.path.exists(vocabulary_path):
        vocab = {}
        with open(data_path,mode='r',encoding='utf-8') as f:
            count = 0
            for line in f:
                line = line.strip()
                count +=1
                if count % 10000 == 0:
                    print("处理了数据：%d" % count)

                tokens = tokenizer(line) if tokenizer else line.split(" ")
                for word in tokens:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab = {key:value for key,value in vocab.items() if value >filter_num}
            vocab_list = add_tokens + sorted(vocab,key=vocab.get,reverse=True)
            print("过滤器，词库大小：%d" % len(vocab_list))

            with open(vocabulary_path,mode='w',encoding='utf-8') as vocab_write:
                for w in vocab_list:
                    vocab_write.write(w + "\n")
    else:
        print("vocab：%s 已经生产" % vocabulary_path)


if __name__ == '__main__':
    add_tokens = ['</S>','<S>','<UNK>']
    create_vocabulary(config.vocabulary_path,config.train_data_path,chinese_tokenizer,add_tokens,config.filter_min_count)