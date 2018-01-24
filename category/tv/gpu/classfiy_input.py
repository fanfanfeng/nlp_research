# create by fanfan on 2018/1/23 0023
from category.tv.gpu import classfy_setting
import tensorflow as tf

def distorted_inputs(data_path, batch_size):
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(data_path)
    reader = tf.TextLineReader()
    key,value = reader.read(filename_queue)

    decoded= tf.decode_csv(value,field_delim=" ",record_defaults=[[0] for i in range(classfy_setting.max_document_length+1)])
    whole_list = tf.train.shuffle_batch(decoded,
                           batch_size= batch_size,
                           capacity= batch_size * 50,
                           min_after_dequeue= batch_size * 10,
                            num_threads=4)
    features = tf.transpose(tf.stack(whole_list[0:classfy_setting.max_document_length]))
    features = tf.reverse(features,axis=[1])
    label = tf.transpose(tf.stack(whole_list[classfy_setting.max_document_length:]))
    label = tf.reshape(label,(batch_size,))
    return features,label


if __name__ == '__main__':
    text_path= r'E:\tv_category\train_and_test\classify_test.txt'
    feature,label = distorted_inputs([text_path],10)
    with tf.Session() as sess:
        # Start populating the filename queue.
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



