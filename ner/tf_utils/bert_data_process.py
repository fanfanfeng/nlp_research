# create by fanfan on 2019/4/15 0015
import os
import random
import tqdm
import json

class NormalData():
    def __init__(self,folder,output_path=None):
        self.folder_path = folder
        self.output_path = output_path

    def load_data(self):
        for folder in os.listdir(self.folder_path):
            itention_path = os.path.join(self.folder_path,folder)
            for file in os.listdir(itention_path):
                with open(os.path.join(itention_path,file), 'r', encoding='utf-8') as fr:
                    for line in fr:
                        yield line.strip().split(" ")

    def getTotalfiles(self):
        total_file = []
        for folder in os.listdir(self.folder_path):
            itention_path = os.path.join(self.folder_path,folder)
            for file in os.listdir(itention_path):
                total_file.append(os.path.join(itention_path,file))
        random.shuffle(total_file)
        return total_file


    def load_single_file(self,file):
        with open(file, 'r', encoding='utf-8') as fr:
            for line in fr:
                tokens = [token for token in line.strip().split(" ") if token != ""]
                yield tokens



    def create_label_dict(self,bert_model_path):
        label_list = set()
        for line  in self.load_data():
            for token in line:
                word_and_type = token.split("\\")
                if len(word_and_type) == 2:
                    label_list.add(word_and_type[1])
        label_list =  ['O']  + list(label_list)


        vocab_list = []
        vocab_path = os.path.join(bert_model_path,'vocab.txt')
        with open(vocab_path,'r',encoding='utf-8') as fread:
            for word in fread:
                vocab_list.append(word.strip())
        vocab_dict = {key:index for index,key in enumerate(vocab_list)}



        if self.output_path != None:
            with open(os.path.join(self.output_path, "vocab.txt"), 'w', encoding='utf-8') as fwrite:
                for word in vocab_list:
                    fwrite.write(word + "\n")

            with open(os.path.join(self.output_path, 'label.txt'), 'w', encoding='utf-8') as fwrite:
                for itent in label_list:
                    fwrite.write(itent + "\n")

        return vocab_dict, vocab_list, label_list

    def load_vocab_and_labels(self):
        vocab_list = []
        labels = []

        with open(os.path.join(self.output_path, "vocab.txt"), 'r', encoding='utf-8') as fread:
            for word in fread:
                vocab_list.append(word.strip())

        with open(os.path.join(self.output_path, 'label.txt'), 'r', encoding='utf-8') as fread:
            for label in fread:
                labels.append(label.strip())

        return {key: index for index, key in enumerate(vocab_list)}, vocab_list, labels

class RasaData():
    def __init__(self,folder,min_freq=1,output_path=None):
        self.folder_path = folder
        self.output_path = output_path
        self.train_folder = os.path.join(self.folder_path, 'train')
        self.test_folder = os.path.join(self.folder_path, 'test')

    def load_folder_data(self,folder_path):
        total_files = self.getTotalfiles(folder_path)

        for file in tqdm.tqdm(total_files):
            with open(file, 'r', encoding='utf-8') as fr:
                data = json.load(fr)
                for item in data['rasa_nlu_data']['common_examples']:
                    entity_offsets = self._convert_example(item)
                    label_list = self._predata(item['text'],entity_offsets)
                    yield list(item['text']), label_list

    def getTotalfiles(self,folder_path):
        files = []
        if os.path.isfile(folder_path):
            files.append(folder_path)
        else:
            for file in os.listdir(folder_path):
                files.append(os.path.join(folder_path, file))
        random.shuffle(files)
        return files





    def create_label_dict(self,bert_model_path):

        label_list = set()
        for line, labels in self.load_folder_data(self.train_folder):
            for word,label in zip(line,labels):
                label_list.add(label)

        for line, labels in self.load_folder_data(self.test_folder):
            for word, label in zip(line, labels):
                label_list.add(label)

        vocab_list = []
        vocab_path = os.path.join(bert_model_path, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as fread:
            for word in fread:
                vocab_list.append(word.strip())
        vocab_dict = {key: index for index, key in enumerate(vocab_list)}

        label_list = list(label_list)
        label_list.remove("O")
        label_list = ["O"]+label_list
        if self.output_path != None:
            with open(os.path.join(self.output_path, "vocab.txt"), 'w', encoding='utf-8') as fwrite:
                for word in vocab_list:
                    fwrite.write(word + "\n")

            with open(os.path.join(self.output_path, 'label.txt'), 'w', encoding='utf-8') as fwrite:
                for itent in label_list:
                    fwrite.write(itent + "\n")

        return vocab_dict, vocab_list, label_list

    def load_vocab_and_labels(self):
        vocab_list = []
        labels = []

        with open(os.path.join(self.output_path, "vocab.txt"), 'r', encoding='utf-8') as fread:
            for word in fread:
                vocab_list.append(word.strip())

        with open(os.path.join(self.output_path, 'label.txt'), 'r', encoding='utf-8') as fread:
            for label in fread:
                labels.append(label.strip())

        return {key: index for index, key in enumerate(vocab_list)}, vocab_list, labels

    @staticmethod
    def _convert_example(example):
        def convert_entity(entity):
            return entity["start"], entity["end"], entity["entity"]

        return [convert_entity(ent) for ent in example.get("entities", [])]

    @staticmethod
    def _predata(text, entity_offsets):
        value = 'O'
        bilou = [value for _ in text]
        for (start, end, entity) in entity_offsets:
            if start is not None and end is not None:
                bilou[start] = 'B-' + entity
                for i in range(start + 1, end):
                    bilou[i] = 'I-' + entity
        return bilou


if __name__ == '__main__':
    normal_data = NormalData(r'E:\qiufengfeng_ubuntu_beifen\word_vec_mode\ner_right')
    #normal_data.create_vocab_dict()