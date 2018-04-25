# -*- coding: utf-8 -*-
# @Author: qiufeng


import os
import os.path

from DualLSTMEncoderRankModel import config
from utils import text_util
import re

def load_no_processed_xiaohuangji(source,target):
    conversation = []
    with open(target,mode='x',encoding='utf-8') as fwrite:
        with open(source, 'r', encoding='utf-8') as f:
            lineID = 0
            label = None
            for line in f:
                label = line[0]
                if label == 'E':

                    conversation = [sen for sen in conversation if sen!=""]
                    if len(conversation) < 2:
                        conversation = []
                        continue
                    else:
                        conversation = conversation[:2]
                        fwrite.write("###".join(conversation) + "\n")
                        conversation = []
                else:
                    if label != 'E':
                        content = line[2:].strip()
                        content = text_util.clean_text(content).strip()
                        conversation.append(content)
                lineID += 1
