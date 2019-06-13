# create by fanfan on 2019/5/24 0024
import sys
import os
import random

class NormalData():
    def __init__(self,folder,min_freq=1,output_path=None):
        self._START_VOCAB = ['_PAD', '_GO', "_EOS", '<UNK>']
        self.folder_path = folder
        self.min_freq = min_freq
        self.output_path = output_path

    def load_data(self):
        for file in self.getTotalfiles():
            yield self.load_single_file(file)

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
                yield line.strip().split('\t')



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

        vocab = {key: value for key, value in vocab.items() if value >= self.min_freq}
        vocab_list = self._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
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