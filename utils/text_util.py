# create by fanfan on 2018/4/16 0016
import jieba
import os
import codecs
def clean_text(in_str):
    out_str=''
    for i in range(len(in_str)):
        if is_uchar(in_str[i]):
            out_str=out_str+in_str[i]
        else:
            out_str=out_str+' '
    return out_str

def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True
    """判断一个unicode是否是英文字母"""
    #if (uchar >= u'\u0041' and uchar<=u'\u005a') or (uchar >= u'\u0061' and uchar<=u'\u007a'):
           # return True
    #if uchar in ('-',',','，','。','.','>','?'):
            #return True

    return False


def is_chinese_uchar(uchar):
    '''
    判断一个unicode是否是汉字
    :param uchar: 
    :return: 
    '''
    if uchar >= u'\u4e00' and uchar<=u'\u9fa5':
            return True
    return False

def is_english_uchar(uchar):
    '''
    判断一个unicode是否是英文字母
    :param uchar: 
    :return: 
    '''
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    return False

def is_digte_uchar(uchar):
    '''
    判断一个unicode是否是数字
    :param uchar: 
    :return: 
    '''
    if uchar >= u'\u0030' and uchar<=u'\u0039':
            return True
    return False


def chinese_tokenizer(document):
    """
    把中文文本转为词序列
    """
    tokens = list(jieba.cut(document))
    tokens = [word for word in tokens if is_chinese_uchar(word)]
    # 分词

    return tokens

import re
def regex_clean_kuohao(text):
    text = re.sub(r'\(.*\)',"",text)
    return text

def regex_clean_shuminghao(text):
    text = re.sub(r'《.*》', "", text)
    return text



stop_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'stopwords.txt')

stop = set()
fr = codecs.open(stop_path, 'r', 'utf-8')
for word in fr:
    stop.add(word.strip())
fr.close()
re_zh = re.compile('([\u4E00-\u9FA5]+)')


def filter_stop(words):
    return list(filter(lambda x: x not in stop, words))

def get_sentences(doc):
    line_break = re.compile('[\r\n]')
    delimiter = re.compile('[，。？！；]')
    sentences = []
    for line in line_break.split(doc):
        line = line.strip()
        if not line:
            continue
        for sent in delimiter.split(line):
            sent = sent.strip()
            if not sent:
                continue
            sentences.append(sent)
    return sentences

if __name__ == '__main__':
    print(clean_text("很无聊哎〜都不知道想干嘛！你在干嘛呢？"))