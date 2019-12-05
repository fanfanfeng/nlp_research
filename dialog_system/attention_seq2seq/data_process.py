# create by fanfan on 2019/5/24 0024
import sys
import os
import random
import re
from utils.langconv import Traditional2Simplified
def find_and_validate(line):
    return ''.join(re.findall(r'[\u4e00-\u9fff]+', line))

import jieba
current_path = os.path.dirname(os.path.abspath(__file__))
jieba.load_userdict(os.path.join(current_path,"user_dict/user_dict.txt"))

def remove_punc(line):
    line = line.replace('。','')
    line = line.replace('？','')
    line = line.replace('！','')
    line = line.replace('，','')
    line = line.replace('.','')
    line = line.replace(',','')
    line = line.replace('?','')
    line = line.replace('!','')
    line = line.replace('“','')
    line = line.replace('”','')
    line = line.replace('¥', '')
    line = line.replace('@', '')
    line = line.replace('\n', '')
    line = line.replace('(', '')
    line = line.replace(')', '')
    return line

class NormalData():
    def __init__(self,folder,min_freq=2,output_path=None,max_vocab_size = 0):
        self._START_VOCAB = ['_PAD', '_GO', '<UNK>',"_EOS"]
        self.folder_path = folder
        self.min_freq = min_freq
        self.output_path = output_path
        self.max_vocab_size = max_vocab_size

    def load_data(self):
        for file in self.getTotalfiles():
            with open(file, 'r', encoding='utf-8') as fr:
                for line in fr:
                    source_and_target= list(line.strip().split('\t'))
                    if len(source_and_target) != 2:
                        continue
                    source,target = source_and_target
                    source = remove_punc(source)
                    target = remove_punc(target)

                    source = Traditional2Simplified(source)
                    target = Traditional2Simplified(target)

                    source = [find_and_validate(word) for word in jieba.cut(source) if find_and_validate(word) != ""]
                    target = [find_and_validate(word) for word in jieba.cut(target) if find_and_validate(word) != ""]
                    if source == [] or target == []:
                        continue
                    yield source,target

    def getTotalfiles(self):
        total_files = []
        for folder in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path,folder)
            total_files.append(file_path)
        random.shuffle(total_files)
        return total_files


    def load_single_file(self,file):
        with open(file, 'r', encoding='utf-8') as fr:
            for line in fr:
                source_and_target = list(line.strip().split('\t'))
                if len(source_and_target) != 2:
                    continue
                source, target = source_and_target

                source = remove_punc(source)
                target = remove_punc(target)

                source = Traditional2Simplified(source)
                target = Traditional2Simplified(target)
                source = [find_and_validate(word) for word in jieba.cut(source) if
                          find_and_validate(word) != ""]
                target = [find_and_validate(word) for word in jieba.cut(target) if
                          find_and_validate(word) != ""]
                yield source,target



    def create_vocab_dict(self):
        vocab = {}
        for source,target in self.load_data():
            for word in source:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

            for word in target:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

        print('过滤词频前，词库大小：%s' % len(vocab.values()))
        vocab = {key: value for key, value in vocab.items() if value >= self.min_freq}
        vocab_list = self._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        print('过滤词频后，词库大小：%s' % len(vocab.values()))

        if  self.max_vocab_size > 0 and len(vocab_list) > self.max_vocab_size:
            vocab_list = vocab_list[:self.max_vocab_size]
        vocab_dict = {key: index for index, key in enumerate(vocab_list)}

        if self.output_path != None:
            with open(os.path.join(self.output_path, "vocab.txt"), 'w', encoding='utf-8') as fwrite:
                for word in vocab_list:
                    fwrite.write(word + "\n")

        return vocab_dict, vocab_list


    def load_vocab_and_intent(self):
        vocab_list = []
        with open(os.path.join(self.output_path, "vocab.txt"), 'r', encoding='utf-8') as fread:
            for word in fread:
                vocab_list.append(word.strip())

        return {key: index for index, key in enumerate(vocab_list)}, vocab_list


