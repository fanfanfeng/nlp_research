# create by fanfan on 2018/1/23 0023
import sys
project_path = r'/data/python_project/nlp_research'
sys.path.append(project_path)
from category.tv.gpu import classfiy_input
from category.tv.gpu.clasfiy_model import Attention_lstm_model
from category.tv.gpu import classfy_setting
import tensorflow as tf
import time
from datetime import datetime
from sklearn.metrics import f1_score


num_gpus = 1
TOWER_NAME = 'tower'
log_device_placement = False
def train():
    with tf.Graph().as_default(),tf.device('/cpu:0'):
        model = Attention_lstm_model()

        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        total_step = classfy_setting.batch_size * classfy_setting.num_epochs
        decay_steps = total_step * classfy_setting.min_learning_rate / classfy_setting.initial_learning_rate
        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(classfy_setting.initial_learning_rate,
                                        global_step,
                                        decay_steps,
                                        classfy_setting.decay_rate,
                                        staircase=True)
        # Create an optimizer that performs gradient descent.
        optimizer = tf.train.GradientDescentOptimizer(lr)

        train_x, train_y = classfiy_input.distorted_inputs([classfy_setting.train_data_path],
                                                           classfy_setting.batch_size)
        test_x, test_y = classfiy_input.distorted_inputs([classfy_setting.test_data_path], classfy_setting.batch_size)
        tower_grads = []
        #with tf.variable_scope("train"):
        for i in range(num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope("%s_%d" % (TOWER_NAME,i)) as scope:
                    loss,_ = model.tower_loss(scope,train_x,train_y)
                    tf.get_variable_scope().reuse_variables()
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

        #with tf.variable_scope("train",reuse=True) as scope:
        tf.get_variable_scope().reuse_variables()
        _, logits = model.tower_loss(scope=None,input_x=test_x,input_y=test_y)
        prediction = tf.cast(tf.argmax(logits, 1, name="prediction"),tf.int32)

        grads = model.average_gradients(tower_grads)
        no_word2vec_grads = [temp_grads for temp_grads in grads if
                             not temp_grads[1].name.startswith('word2vec_embedding')]
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
        train_op_no_word2vec = optimizer.apply_gradients(no_word2vec_grads, global_step=global_step)
        saver = tf.train.Saver(tf.global_variables())

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=log_device_placement
        ), graph=tf.get_default_graph())


        model.restore_model(sess,saver)
        tf.train.start_queue_runners(sess=sess)

        best_f1 = 0
        for step in range(total_step):
            print("step  {}".format(step + 1))
            start_time = time.time()
            feed_dict = {model.dropout:classfy_setting.dropout}
            if step < int(total_step*0.7):
                _, loss_value = sess.run([train_op_no_word2vec, loss], feed_dict)
            else:
                _,loss_value= sess.run([train_op,loss],feed_dict)
            duration = time.time() - start_time
            if (step+1) % classfy_setting.show_every == 0:
                num_example_per_step = classfy_setting.batch_size * num_gpus
                examples_per_sec = num_example_per_step / duration
                sec_per_batch = duration / num_gpus
                format_str = "%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec /batch)"
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

                #graph_writer.add_summary(summary_op, step)
            if (step+1) % classfy_setting.valid_every == 0  :
                feed_dict = {model.dropout:1.0}
                prediction_total = []
                target_total = []
                for i in range(60):
                    prediction_val ,target_val= sess.run([prediction,test_y],feed_dict=feed_dict)
                    prediction_total.extend(prediction_val)
                    target_total.extend(target_val)
                f1 = f1_score(target_total,prediction_total,labels=[1,2,3,4],average='micro')
                print("验证模型, 训练步数 {} , f值 {:g}".format(step, f1))
                if best_f1 < f1:
                    path = saver.save(sess, classfy_setting.train_model_bi_lstm, step)
                    print("模型保存到{}".format(path))
                    best_f1 = f1



if __name__ == '__main__':
    train()