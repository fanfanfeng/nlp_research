# create by fanfan on 2017/12/28 0028
from ner.tv import ner_setting
import os
training_data_path = ner_setting.tv_data_path

def changefile_to_training_format(file_path):
    new_file_path = file_path.replace(".txt","_train.txt")
    with open(new_file_path,'w',encoding='utf-8') as fout:
        with open(file_path,'r',encoding='utf-8') as fin:
            for line in fin:
                tokens = line.strip().split(' ')
                new_list = []
                for token in tokens:
                    single_words_list = token.split("\\")
                    if len(single_words_list) == 1:
                        single_words_list.append("O")
                    elif len(single_words_list) !=2:
                        continue
                    new_list.append("/".join(single_words_list))

                fout.write(' '.join(new_list) + '\n')

    os.remove(file_path)






for root,dirs,files in  os.walk(training_data_path):
    if files != None:
        for file in files:
            real_file_path = os.path.join(root,file)
            if "_train.txt" in real_file_path:
                continue
            changefile_to_training_format(real_file_path)

