__author__ = 'fanfan'
# coding:utf-8
import sys
_PAD = '_PAD'
_GO = '_GO'
_EOS = "_EOS"
_UNK = "_UNK"

class WordToken(object):
    def __init__(self):
        # 最小起始id号, 保留的用于表示特殊标记
        self.START_ID = 4
        self.PAD_ID = 0
        self.GO_ID = 1
        self.EOS_ID = 2
        self.UNK_ID = 3
        self._START_VOCAB = [_PAD,_GO,_EOS,_UNK]
        self.word2id_dict = {}
        self.id2word_dict = {}


    def load_file_list(self, file_list, min_freq):
        """
        加载样本文件列表，全部切词后统计词频，按词频由高到低排序后顺次编号
        并存到self.word2id_dict和self.id2word_dict中
        """
        words_count = {}
        for file in file_list:
            with open(file, 'r',encoding='utf-8') as file_object:
                for line in file_object.readlines():
                    line = line.strip()
                    seg_list = list(line)
                    for str in seg_list:
                        if str in words_count:
                            words_count[str] = words_count[str] + 1
                        else:
                            words_count[str] = 1

        sorted_list = [ v[0] for v in words_count.items() if v[1] > min_freq ]
        sorted_list.sort(reverse=True)
        sorted_list = self._START_VOCAB + sorted_list
        for index, item in enumerate(sorted_list):

            self.word2id_dict[item] = index
            self.id2word_dict[index] = item

        return index

    def word2id(self, word):
        if word in self.word2id_dict:
            return self.word2id_dict[word]
        else:
            return self.UNK_ID

    def id2word(self, id):
        id = int(id)
        if id in self.id2word_dict:
            return self.id2word_dict[id]
        else:
            return _UNK