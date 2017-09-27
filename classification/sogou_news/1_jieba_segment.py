__author__ = 'fanfan'
import os
import sys
data_path = 'data/sogou_data'
data_process_path = 'data/processed'
import jieba

def prepare_data():
    jieba.load_userdict('dict_modify.txt')
    if not os.path.exists(data_process_path):
        os.mkdir(data_process_path)

    dirs = os.listdir(data_path)
    for dir in dirs:
        current_dir = os.path.join(data_path,dir)
        if not os.path.isdir(current_dir):
            continue
        new_dir = current_dir.replace('sogou_data','processed')
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        new_file_writer = open(os.path.join(new_dir,"jieba.txt"),'w',encoding='utf-8')
        for file in os.listdir(current_dir):
            current_path= os.path.join(current_dir,file)
            if os.path.isdir(current_path):
                continue
            with open(current_path,encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line == "":
                        continue
                    tokens = jieba.cut(line)
                    new_file_writer.write(" ".join(list(tokens)) + "\n")
        new_file_writer.close()






if __name__ == '__main__':
    prepare_data()