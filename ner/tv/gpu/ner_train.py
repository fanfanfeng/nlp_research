# create by fanfan on 2018/1/25 0025
import sys
project_path = r'/data/python_project/nlp_research'
sys.path.append(project_path)
import tensorflow as tf
from ner.tv.gpu import ner_input
from ner.tv.gpu import ner_setting
from ner.tv.gpu import ner_model
import time
from datetime import datetime
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import numpy as np
import os
from ner.tv.gpu import data_util


num_gpus = 1
TOWER_NAME = 'tower'
log_device_placement = False


def train():
    with tf.Graph().as_default(),tf.device("/cpu:0"):
        model = ner_model.Model()
        global_step = tf.Variable(0,trainable=False,name='global_step')
        lr = tf.train.exponential_decay(ner_setting.initial_learning_rate,global_step=global_step,decay_steps=ner_setting.decay_step,decay_rate=ner_setting.decay_rate,staircase=True)
        lr = tf.maximum(lr,ner_setting.min_learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        train_x,train_y = ner_input.distorted_inputs([ner_setting.train_data_path],ner_setting.batch_size)

        tower_grads = []
        for i in range(num_gpus):
            with tf.name_scope('{}_{}'.format(TOWER_NAME,i)) as scope:
                loss_train,_,_,_ = model.logits_and_loss(train_x,train_y)
                grads_train = optimizer.compute_gradients(loss_train)
                #tf.get_variable_scope().reuse_variables()
                tower_grads.append(grads_train)
        grads_train = model.average_gradients(tower_grads)
        no_word2vec_grads = [temp_grad for temp_grad in grads_train if  not temp_grad[1].name.startswith("word2vec_embedding") ]

        train_op = optimizer.apply_gradients(grads_train,global_step=global_step)
        train_op_no_word2vec = optimizer.apply_gradients(no_word2vec_grads,global_step=global_step)

        tf.get_variable_scope().reuse_variables()
        _, logits_test, length_test, trans_matrix_test = model.logits_and_loss(input_x=model.inputs, input_y=model.labels)

        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=log_device_placement
        ))
        model.model_restore(sess, saver)

        tf.train.start_queue_runners(sess=sess)
        best_f1 = 0
        total_step = ner_setting.num_epochs * ner_setting.step_per_epochs
        print("开始训练")
        test_manager = data_util.BatchManagerTest(ner_setting.test_data_path, ner_setting.batch_size)
        for epoch in range(total_step):
            start_time = time.time()
            train_feed_dict = {model.dropout:ner_setting.dropout}
            if epoch < int(total_step*0.5):
                _, loss_value = sess.run([train_op_no_word2vec, loss_train], train_feed_dict)
            else:
                _,loss_value= sess.run([train_op,loss_train],train_feed_dict)
            duration = time.time() - start_time

            if (epoch+1) % ner_setting.show_ervery == 0:
                num_example_per_step = ner_setting.batch_size * num_gpus
                examples_per_sec = num_example_per_step / duration
                sec_per_batch = duration / num_gpus
                format_str = "%s: step %d,learnging rate %s, loss = %.2f (%.1f examples/sec; %.3f sec /batch)"
                print(format_str % (datetime.now(), epoch + 1, lr.eval(session=sess), loss_value, examples_per_sec, sec_per_batch))

            if ( epoch+1) % ner_setting.checkpoint_every == 0  :
                real_total_labels = []
                predict_total_labels = []
                for test_x,test_y in test_manager.iterbatch():
                    feed_dict = {model.dropout: 1.0, model.inputs:test_x}
                    logits_test_var, lengths_test_var, trans_matrix = sess.run(
                        [logits_test, length_test, trans_matrix_test], feed_dict=feed_dict)
                    real_labels, predict_labels = model.test_accuraty(lengths_test_var, logits_test_var,
                                                                          trans_matrix, test_y)
                    real_total_labels.extend(real_labels)
                    predict_total_labels.extend(predict_labels)
                print(classification_report(real_total_labels, predict_total_labels, labels=list(np.arange(17)),target_names=ner_setting.tagnames))
                f1_score_value = f1_score(real_total_labels, predict_total_labels, labels=list(np.arange(17)),average='micro')
                print("iteration:{},NER ,precition score:{:>9.6f}".format(epoch, f1_score_value))
                if best_f1 < f1_score_value:
                    print("mew best f1_score,save model ")
                    saver.save(sess, model.model_save_path, global_step=epoch)
                    best_f1 = f1_score_value

    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement = True,
            log_device_placement = log_device_placement,
        )) as sess:
        model = ner_model.Model()
        _, logits_out,length_out,trans_matrix_out =  model.logits_and_loss( input_x=model.inputs, input_y=model.labels)
        saver = tf.train.Saver(tf.global_variables())
        model.model_restore(sess,saver)
        output_tensor = []
        output_tensor.append(trans_matrix_out.name.replace(":0", ""))
        output_tensor.append(length_out.name.replace(":0", ""))
        output_tensor.append(logits_out.name.replace(":0", ""))
        output_tensor.append(model.dropout.name.replace(":0", ""))
        output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_tensor)
        with tf.gfile.FastGFile(os.path.join(ner_setting.graph_model_bi_lstm, "weight_ner.pb"),'wb') as gf:
            gf.write(output_graph_with_weight.SerializeToString())

if __name__ == '__main__':
    train()



