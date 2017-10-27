__author__ = 'fanfan'
import pickle
import tensorflow as tf
import os

DATA_PATH = 'data'
CIFAR_LOCAL_FOLDER = "cifar-10-python"
def _get_file_names():
    """Returns the file names expected to exist in the input_dir."""
    file_names = {}
    file_names['train'] = ["data_batch_%d" % i for i in range(1,5)]
    file_names['validation'] = ['data_batch_5']
    file_names['eval'] = ['test_batch']
    return  file_names

def _int64_feature(value):
    return tf.train.Feature(int64_list = tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list = tf.train.BytesList(value=[value]))

def read_pickle_from_file(filename):
    with tf.gfile.Open(filename,'rb') as f:
        data_dict = pickle.load(f,encoding='bytes')
    return  data_dict

def convert_to_tfrecord(input_files,output_file):
    ''' Converts a file to TFRecords'''
    print("Generating %s" % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        for input_file in input_files:
            data_dict = read_pickle_from_file(input_file)
            data = data_dict[b'data']
            labels = data_dict[b'labels']
            num_entries_in_batch = len(labels)
            for i in range(num_entries_in_batch):
                example = tf.train.Example(features=tf.train.Features(
                    feature = {
                        'image':_bytes_feature(data[i].tobytes()),
                        'label':_int64_feature(labels[i])
                    }
                ))
                record_writer.write(example.SerializeToString())

def main(data_dir):
    file_names = _get_file_names()
    input_dir = os.path.join(data_dir,CIFAR_LOCAL_FOLDER)
    for mode,files in file_names.items():
        input_files = [os.path.join(input_dir,f) for f in files]
        output_file = os.path.join(data_dir,mode+".tfrecords")
        try:
            os.remove(output_file)
        except OSError:
            pass
        convert_to_tfrecord(input_files,output_file)
    print("Done!")

if __name__ == '__main__':
    main(DATA_PATH)
