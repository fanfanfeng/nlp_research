# create by fanfan on 2018/2/6 0006
_PAD = '_PAD'
_GO = '_GO'
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD,_GO,_EOS,_UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

from seq2seq.seq2seq_gpu import config
import tensorflow as tf

def distorted_inputs(data_path, batch_size):
    file_queue = tf.train.string_input_producer(data_path)
    reader = tf.TextLineReader()
    key,value = reader.read(file_queue)

    decoded = tf.decode_csv(value,record_defaults=[[0] for i in range(ner_setting.max_document_length * 2)],field_delim=" ")
    whole_list = tf.train.shuffle_batch(decoded,
                                        batch_size=batch_size,
                                        capacity=batch_size * 50,
                                        min_after_dequeue=batch_size * 10,
                                        num_threads=4)
    features = tf.transpose(tf.stack(whole_list[0:ner_setting.max_document_length]))
    label = tf.transpose(tf.stack(whole_list[ner_setting.max_document_length:]))
    return features,label


if __name__ == '__main__':
    text_path = ner_setting.test_data_path
    feature, label = distorted_inputs([text_path], 3)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            # while not coord.should_stop():
            while True:
                feature_val,label_val = sess.run([feature,label])
                print(feature_val)
                break

        except tf.errors.OutOfRangeError:
            print('Done reading')
        finally:
            coord.request_stop()

        coord.join(threads)
