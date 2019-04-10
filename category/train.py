# create by fanfan on 2019/4/10 0010
import tensorflow as tf
from utils.tfrecord_api import _int64_feature
from category.data_utils import pad_sentence,create_vocab_dict,load_vocab_and_intent,load_rasa_data
from category.models.classify_cnn_model import ClassifyCnnModel,CNNConfig
import os

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"output")
if not os.path.exists(output_path):
    os.mkdir(output_path)



def make_tfrecord_files(file_or_folder,classify_config):

    if os.path.exists(os.path.join(output_path,'vocab.txt')):
        vocab,vocab_list,intent = load_vocab_and_intent(output_path)
    else:
        vocab,vocab_list,intent = create_vocab_dict(file_or_folder,output_path=output_path)

    intent_ids = {key:index for index,key in enumerate(intent)}
    # tfrecore 文件写入
    tfrecord_save_path = os.path.join(classify_config.save_path,"train.tfrecord")
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_save_path)

    files = []
    if os.path.isfile(file_or_folder):
        files.append(file_or_folder)
    else:
        for file in os.listdir(file_or_folder):
            files.append(os.path.join(file_or_folder, file))

    for file in files:
        sentences,intentions = load_rasa_data(file)
        for sentence,intent in zip(sentences,intentions):
            sentence_ids = pad_sentence(sentence,classify_config.max_sentence_length,vocab)
            #sentence_ids_string = np.array(sentence_ids).tostring()
            train_feature_item = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(intent_ids[intent]),
                'sentence':_int64_feature(sentence_ids,need_list=False)
            }))
            tfrecord_writer.write(train_feature_item.SerializeToString())
    tfrecord_writer.close()


if __name__ == '__main__':
    classify_config = CNNConfig()
    classify_config.save_path = output_path
    make_tfrecord_files(r'E:\nlp-data\rasa_corpose',classify_config)
