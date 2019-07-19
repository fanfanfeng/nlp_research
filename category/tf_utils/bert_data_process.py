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
                        yield line.strip().split(" "),folder

    def getTotalfiles(self):
        total_file_and_itent = []
        for folder in os.listdir(self.folder_path):
            itention_path = os.path.join(self.folder_path, folder)
            for file in os.listdir(itention_path):
                total_file_and_itent.append((os.path.join(itention_path, file), folder))
        random.shuffle(total_file_and_itent)
        return total_file_and_itent


    def load_single_file(self,file):
        with open(file, 'r', encoding='utf-8') as fr:
            for line in fr:
                yield line.strip().split(" "),""



    def create_label_dict(self,bert_model_path):
        label_list = set()
        for file,intent  in self.getTotalfiles():
            label_list.add(intent)


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
    def __init__(self,folder,output_path=None):
        self.folder_path = folder
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
                    yield list(item['text']), item['intent']

    def create_label_dict(self,bert_model_path):
        intent_list = set()

        for line, intent in self.load_folder_data(self.test_folder):
            intent_list.add(intent)


        intent = list(intent_list)

        vocab_list = []
        vocab_path = os.path.join(bert_model_path, 'vocab.txt')
        with open(vocab_path, 'r', encoding='utf-8') as fread:
            for word in fread:
                vocab_list.append(word.strip())
        vocab_dict = {key: index for index, key in enumerate(vocab_list)}

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
                yield list(item['text']), item['intent']


if __name__ == '__main__':
    normal_data = NormalData(r'E:\qiufengfeng_ubuntu_beifen\word_vec_mode\ner_right')
    #normal_data.create_vocab_dict()