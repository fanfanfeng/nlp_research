# create by fanfan on 2019/4/15 0015
import os
import json
import tqdm
import jieba
import random
import sys


if 'win' in sys.platform:
    user_dict_path = r'E:\nlp-data\jieba_dict\dict_modify.txt'
else:
    user_dict_path = r'/data/python_project/rasa_corpus/jieba_dict/dict_modify.txt'

jieba.load_userdict(user_dict_path)

class NormalData():
    def __init__(self,folder,min_freq=3,output_path=None):
        self._START_VOCAB = ['_PAD', '_GO', "_EOS", '<UNK>']
        self.folder_path = folder
        self.min_freq = min_freq
        self.output_path = output_path

    def load_data(self):
        for folder in os.listdir(self.folder_path):
            itention_path = os.path.join(self.folder_path,folder)
            for file in os.listdir(itention_path):
                with open(os.path.join(itention_path,file), 'r', encoding='utf-8') as fr:
                    for line in fr:
                        yield line.strip().split(" "),folder

    def getTotalfiles(self):
        total_file_and_itent = []
        for folder in os.listdir(self.folder_path):
            itention_path = os.path.join(self.folder_path,folder)
            for file in os.listdir(itention_path):
                total_file_and_itent.append((os.path.join(itention_path,file),folder))
        random.shuffle(total_file_and_itent)
        return total_file_and_itent


    def load_single_file(self,file):
        with open(file, 'r', encoding='utf-8') as fr:
            for line in fr:
                yield line.strip().split(" "),""



    def create_vocab_dict(self):
        vocab = {}
        intent_list = set()
        for line ,intent in self.load_data():
            for word in line:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1

            intent_list.add(intent)
        vocab = {key: value for key, value in vocab.items() if value >= self.min_freq}
        vocab_list = self._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        vocab_dict = {key: index for index, key in enumerate(vocab_list)}
        intent = list(intent_list)

        if self.output_path != None:
            with open(os.path.join(self.output_path, "vocab.txt"), 'w', encoding='utf-8') as fwrite:
                for word in vocab_list:
                    fwrite.write(word + "\n")

            with open(os.path.join(self.output_path, 'label.txt'), 'w', encoding='utf-8') as fwrite:
                for itent in intent:
                    fwrite.write(itent + "\n")

        return vocab_dict, vocab_list, intent

    def load_vocab_and_intent(self):
        vocab_list = []
        intent = []

        with open(os.path.join(self.output_path, "vocab.txt"), 'r', encoding='utf-8') as fread:
            for word in fread:
                vocab_list.append(word.strip())

        with open(os.path.join(self.output_path, 'label.txt'), 'r', encoding='utf-8') as fread:
            for itent in fread:
                intent.append(itent.strip())

        return {key: index for index, key in enumerate(vocab_list)}, vocab_list, intent


class RasaData():
    def __init__(self,folder,min_freq=3,output_path=None):
        self._START_VOCAB = ['_PAD', '_GO', "_EOS", '<UNK>']
        self.folder_path = folder
        self.min_freq = min_freq
        self.output_path = output_path

        self.train_folder = os.path.join(self.folder_path,'train')
        self.test_folder = os.path.join(self.folder_path,'test')

    def load_folder_data(self,folder_path):

        files = []
        if os.path.isfile(folder_path):
            files.append(folder_path)
        else:
            for file in os.listdir(folder_path):
                files.append(os.path.join(folder_path, file))

        for file in tqdm.tqdm(files):
            with open(file, 'r', encoding='utf-8') as fr:
                data = json.load(fr)
                for item in data['rasa_nlu_data']['common_examples']:
                    yield list(jieba.cut(item['text'])), item['intent']

    def create_vocab_dict(self):
        vocab = {}
        intent_list = set()
        for line, intent in self.load_folder_data(self.train_folder):
            for word in line:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            intent_list.add(intent)

        for line, intent in self.load_folder_data(self.test_folder):
            for word in line:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
            intent_list.add(intent)

        vocab = {key: value for key, value in vocab.items() if value >= self.min_freq}
        vocab_list = self._START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        vocab_dict = {key: index for index, key in enumerate(vocab_list)}
        intent = list(intent_list)

        if self.output_path != None:
            with open(os.path.join(self.output_path, "vocab.txt"), 'w', encoding='utf-8') as fwrite:
                for word in vocab_list:
                    fwrite.write(word + "\n")

            with open(os.path.join(self.output_path, 'label.txt'), 'w', encoding='utf-8') as fwrite:
                for itent in intent:
                    fwrite.write(itent + "\n")

        return vocab_dict, vocab_list, intent

    def load_vocab_and_intent(self):
        vocab_list = []
        intent = []

        with open(os.path.join(self.output_path, "vocab.txt"), 'r', encoding='utf-8') as fread:
            for word in fread:
                vocab_list.append(word.strip())

        with open(os.path.join(self.output_path, 'label.txt'), 'r', encoding='utf-8') as fread:
            for itent in fread:
                intent.append(itent.strip())

        return {key: index for index, key in enumerate(vocab_list)}, vocab_list, intent


    def getTotalfiles(self):
        total_file_and_itent = []
        if os.path.isfile(self.folder_path):
            total_file_and_itent.append((self.folder_path,""))
        else:
            for file in os.listdir(self.folder_path):
                total_file_and_itent.append((os.path.join(self.folder_path, file),""))
        return total_file_and_itent


    def load_single_file(self,file):
        with open(file, 'r', encoding='utf-8') as fr:
            data = json.load(fr)
            for item in data['rasa_nlu_data']['common_examples']:
                yield list(jieba.cut(item['text'])), item['intent']

if __name__ == '__main__':
    normal_data = RasaData(r'E:\nlp-data\rasa_corpose',output_path=r'E:\git-project\nlp_research\category\output')
    normal_data.create_vocab_dict()