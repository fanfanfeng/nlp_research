# create by fanfan on 2020/3/13 0013
# coding=utf-8
import json
import csv
from Competition.datafountain_emotion import settings
from utils.common import puncs
from utils.text_util import filter_stop
import jieba

def make_train_and_dev():
    train_list = []
    with open(settings.train_origin_data_path, 'r',encoding='utf-8') as fread:
            for line in fread:
                weiboId, weiboTime, userid, text, img, video, label = next(csv.reader(line.splitlines(), skipinitialspace=True))
                if label not in settings.label_list:
                    continue
                train_list.append([weiboId,drop_stop_words(text), label])

    train_end = int(len(train_list) * 0.9)
    dev_end = int(len(train_list))

    with open(settings.train_data_path, "w",encoding='utf-8') as fwrite:
        for i in range(0, train_end):
            fwrite.write("\t".join(train_list[i]) + "\n")

    with open(settings.dev_data_path, "w",encoding='utf-8') as fwrite:
        for i in range(train_end + 1, dev_end):
            fwrite.write("\t".join(train_list[i]) + "\n")

def drop_stop_words(text):
    tokens = jieba.cut(text)
    output = filter_stop(list(tokens))
    return "".join(output)

def make_test():
    test_list = []
    with open(settings.test_origin_data_path, 'r',encoding='utf-8') as fread:
        for index, line in enumerate(fread):
            if index == 0:
                continue

            weiboId, weiboTime, userid, text, img, video = next(csv.reader(line.splitlines(), skipinitialspace=True))
            test_list.append([weiboId,text])

    with open(settings.test_data_path, "w",encoding='utf-8') as fwriter:
        for i in range(len(test_list)):
            fwriter.write("\t".join(test_list[i]) + "\n")

if __name__ == '__main__':
    make_train_and_dev()
    make_test()