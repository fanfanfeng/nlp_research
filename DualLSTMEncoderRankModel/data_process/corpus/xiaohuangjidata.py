# -*- coding: utf-8 -*-
# @Author: qiufeng


import os
import os.path
import jieba

from DualLSTMEncoderRankModel import config

"""
Opensource Xiaohuangji Dialogue Corpus

"""

class XiaohuangjiData:
    """
    """
    def __init__(self, dirName):
        """
        Args:
            dirName (string): data directory of xhj data
        """

        if os.path.isfile(os.path.join(dirName, 'xhj.pkl')):
            print('loading from xhj.pkl')
            import pickle
            with open(os.path.join(dirName, 'xhj.pkl'),'rb') as f:
                self.conversations = pickle.load(f)
        else:
            self.conversations = []
            fileName = os.path.join(dirName, 'xiaohuangji50w_nofenci.conv')
            # fileName = os.path.join(dirName, 'test.conv')
            self.loadConversations(fileName)


    def loadConversations(self, fileName):
        """
        Args:
            fileName (str): file to load
        Return:
            list<dict<str>>: the extracted fields for each line
        """
        with open(fileName, 'r',encoding='utf-8') as f:
            lineID = 0
            label= None
            for line in f:
                if lineID<100:
                    print(line)
                if lineID==0 or label=='E': # next dialogue
                    label = line[0]
                    content = line[2:].strip()
                    content = self.segment(content)
                    conversation = [{"text": [content.split('/')]}]
                else:
                    label = line[0]
                    if label!='E':
                        content = line[2:].strip()
                        content = self.segment(content)
                        conversation.append({"text":[content.split('/')]})
                    else:
                        self.conversations.append({"lines":conversation})
                lineID += 1
        return self.conversations


    def getConversations(self):
        return self.conversations

    def segment(self, content):
        seg_list = jieba.cut(content, cut_all=False)
        return "/".join(seg_list)

