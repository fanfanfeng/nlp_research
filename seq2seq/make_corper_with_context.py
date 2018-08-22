# create by fanfan on 2018/3/12 0012
import os
from collections import deque
_PAD = '_PAD'
origin_data_path = 'seq2seq_dynamic/data'
out_file = os.path.join(origin_data_path,'data_with_context.txt')
out_write = open(out_file,'w',encoding='utf-8')

context_one = ""
context_two = ""
ask = ""
answer = ""

deque_obj = deque(maxlen=4)
deque_obj.append(context_one)
deque_obj.append(context_two)
deque_obj.append(ask)
deque_obj.append(answer)


def make_context_text(contex_one,context_two,sentence):
    total_list = []
    context_one_list = contex_one.strip().split(" ")
    context_one_list = [item for item in context_one_list if item!=""]
    if len(context_one_list) <= 10:
        context_one_list = context_one_list + ( 10 - len(context_one_list))*[_PAD]
    else:
        context_one_list = context_one_list[:10]
    total_list += context_one_list

    context_two_list = context_two.strip().split(" ")
    context_two_list = [item for item in context_two_list if item != ""]
    if len(context_two_list) <= 10:
        context_two_list = context_two_list + (10 - len(context_two_list)) * [_PAD]
    else:
        context_two_list = context_two_list[:10]
    total_list += context_two_list

    sentence_list = sentence.strip().split(" ")
    sentence_list = [item for item in sentence_list if item != ""]
    #if len(sentence_list) <= 10:
        #sentence_list = sentence_list + (10 - len(sentence_list)) * [_PAD]
    #else:
        #sentence_list = sentence_list[:10]
    total_list += sentence_list

    return " ".join(total_list)

with open(os.path.join(origin_data_path,'train.txt'),encoding='utf-8') as fread:
    for index, line in enumerate(fread):
        deque_obj.append(line)
        print(index)
        if index < 1:
            continue
        context_ask = deque_obj[-2]#make_context_text(*list(deque_obj)[:3])
        context_answer = deque_obj[-1]

        out_write.write(context_ask)
        out_write.write(context_answer)
out_write.close()

