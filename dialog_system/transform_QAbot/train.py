# create by fanfan on 2019/5/24 0024
# create by fanfan on 2019/4/10 0010
import sys
sys.path.append(r"/data/python_project/nlp_research")
import tensorflow as tf
from utils.tfrecord_api import _int64_feature
import os
import tqdm
import argparse
from dialog_system.attention_QAbot import  data_process
from dialog_system.attention_QAbot.data_utils import input_fn,make_tfrecord_files,pad_sentence

def argument_parser():
    parser = argparse.ArgumentParser(description="训练参数")
    parser.add_argument('--output_path', type=str, default='output/', help="中间文件生成目录")
    parser.add_argument('--origin_data', type=str, default='data/', help="原始数据地址")
    parser.add_argument('--device_map', type=str, default="0", help="gpu 的设备id")
    parser.add_argument('--max_sentence_len', type=int, default=50, help='一句话的最大长度')
    return parser.parse_args()

def train(arguments):
    data_processer = data_process.NormalData(arguments.origin_data, output_path=arguments.output_path)



if __name__ == '__main__':
    argument_dict = argument_parser()
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), argument_dict.output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    argument_dict.output_path = output_path