# create by fanfan on 2017/10/18 0018
from HAN_network.Chinese_Sentiment_Classification import settings
from HAN_network.Chinese_Sentiment_Classification import model
from HAN_network.Chinese_Sentiment_Classification import data_util
import time
import tensorflow as tf
import os


def read_tfrecords(index=0):
    train_path = os.path.join(settings.data_dir, 'train.tfrecords')
    valid_path = os.path.join(settings.data_dir, 'valid.tfrecords')
    train_queue = tf.train.string_input_producer([train_path])
    valid_queue = tf.train.string_input_producer([valid_path])

    queue = tf.QueueBase.from_list(index,[train_queue,valid_queue])
    reader = tf.TFRecordReader()
    key,serialized_example = reader.read(queue)
    features = tf.parse_single_example(
        serialized_example,
        features = {
            'sentence_lengths':tf.FixedLenFeature([settings.max_doc_len],tf.int64),
            'document_lengths':tf.FixedLenFeature([],tf.int64),
            'label':tf.FixedLenFeature([],tf.int64),
            'text':tf.FixedLenFeature([settings.max_doc_len * settings.max_sentence_len],tf.int64)
        })
    sentence_lengths = features['sentence_lengths']
    document_lengths = features['document_lengths']
    label = features['label']
    text = features['text']

    sentence_lengths_batch,document_lengths_batch,label_batch,text_batch = tf.train.shuffle_batch(
        [sentence_lengths,document_lengths,label,text],
        batch_size=settings.batch_size,
        capacity=5000,
        min_after_dequeue=1000
    )
    return sentence_lengths_batch,document_lengths_batch,label_batch,text_batch

def main():
    if not os.path.exists(settings.model_save_dir):
        os.makedirs(settings.model_save_dir)

    sentence_lengths_batch,document_lengths_batch,label_batch,text_batch = read_tfrecords(0)
    valid_sentence_lengths_batch,valid_document_lengths_batch,valid_label_batch,valid_text_batch = read_tfrecords(1)

    text_batch = tf.reshape(text_batch,(-1,settings.max_doc_len,settings.max_sentence_len))
    valid_text_batch = tf.reshape(valid_text_batch,(-1,settings.max_doc_len,settings.max_sentence_len))

    _,char_list = data_util.get_vocab()

    model_obj = model.Model()
    with tf.Session() as sess:
        model_obj.restore_model(sess)

        tensorboard_writer = tf.summary.FileWriter(settings.graph_log_path)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            while not coord.should_stop():
                start_time = time.time()
                current_step = model_obj.global_step.eval() + 1
                if current_step % 500 == 0 :
                    valid_cost = 0
                    valid_accuracy = 0
                    for _ in range(50):
                        valid_text, valid_label, valid_sentence_lengths, valid_document_lengths = sess.run([valid_text_batch, valid_label_batch, valid_sentence_lengths_batch, valid_document_lengths_batch])
                        feed_dict={
                            model_obj.inputs:valid_text,
                            model_obj.labels:valid_label,
                            model_obj.sentence_lengths:valid_sentence_lengths,
                            model_obj.document_lengths:valid_document_lengths,
                            model_obj.is_training:False
                        }
                        fetch_list = [model_obj.loss,model_obj.accuracy]

                        valid_outputs = sess.run(fetch_list,feed_dict)
                        valid_cost += valid_outputs[0]
                        valid_accuracy += valid_outputs[1]

                    print("训练第{}步，测试集上面的精确度是{}，loss是{}".format(current_step,valid_accuracy,valid_cost))

                inputs, labels, sentence_lengths, document_lengths = sess.run([text_batch, label_batch, sentence_lengths_batch, document_lengths_batch])
                feed_dict = {
                    model_obj.inputs: inputs,
                    model_obj.labels: labels,
                    model_obj.sentence_lengths: sentence_lengths,
                    model_obj.document_lengths: document_lengths,
                    model_obj.is_training: True
                }
                fetch_list = [model_obj.loss,model_obj.accuracy,model_obj.training_op,model_obj.summary_op]
                train_loss,train_accuracy,_ ,summary= sess.run(fetch_list,feed_dict)
                #写入文件中去
                tensorboard_writer.add_summary(summary,current_step)

                if  current_step % 1 == 0:
                    print("训练第{}步，训练loss{}，精度{}".format(current_step,train_loss,train_accuracy))

                if current_step % 1000 == 0:
                    save_path = model_obj.saver(sess,model_obj.model_save_path)
                    print("保存模型到：",save_path)
        except tf.errors.OutOfRangeError:
            print('Done training!')
        finally:
            coord.request_stop()
        save_path = model_obj.saver(sess, model_obj.model_save_path)
        print("保存模型到：", save_path)
        coord.join(threads)






if __name__ == '__main__':
    main()
