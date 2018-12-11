# create by fanfan on 2018/11/23 0023
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

train_data_path = os.path.join(PROJECT_ROOT,'data/train_data_demo')
filter_min_count = 0 # 词库频率出现的最少次数

vocabulary_path = os.path.join(PROJECT_ROOT,'result/vocab.txt')