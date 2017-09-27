# create by fanfan on 2017/8/26 0026
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import bi_lstm_setting
import  bi_lstm_model
import data_util
import numpy as np
import time
import math

def test():
    model = bi_lstm_model.Model()

def train():
    batch_manager = data_util.BatchManager(bi_lstm_setting.data_processed_path, bi_lstm_setting.batch_size)


    with tf.Session() as sess:
        graph_writer = tf.summary.FileWriter(bi_lstm_setting.graph_save_path,graph=sess.graph)
        model_obj = bi_lstm_model.Model()
        model_obj.restore_model(sess)

        print("开始训练")
        loss = 0.0
        start_time = time.time()
        for epoch_id  in range(bi_lstm_setting.num_epochs):
            for step, (input_x,input_y) in enumerate(batch_manager.train_iterbatch()):
                print("开始第%d轮第%d次训练,globa_step %d" % (epoch_id+1,step+1,model_obj.global_step.eval()))

                # Execute a single training step
                step_loss,_,summary = model_obj.train_step(sess,input_x,input_y,True)
                loss += float(step_loss) / bi_lstm_setting.show_every

                if (model_obj.global_step.eval() +1) % bi_lstm_setting.show_every == 0:
                    avg_perplexity = math.exp(float(loss)) if loss <300 else float("inf")

                    #计算时间
                    time_cost = time.time() - start_time
                    step_time = time_cost / bi_lstm_setting.show_every
                    print('第%d轮训练，第%d的步，loss值为 %.2f , Preplexity值为 %.2f,花费时间 %f' % (epoch_id, model_obj.global_step.eval(),loss,avg_perplexity,step_time))
                    loss = 0.0
                    start_time = time.time()

                    # Record training summary for the current batch
                    graph_writer.add_summary(summary, model_obj.global_step.eval())

                #验证模型
                if (model_obj.global_step.eval() +1) % bi_lstm_setting.valid_every == 0:
                    print("验证模型。。。。。")
                    valid_loss = 0.0
                    valid_accuracy = 0.0
                    test_epoch = batch_manager.test_input_x.shape[0]//batch_manager.batch_size
                    for i in range(test_epoch):
                        test_input_x = batch_manager.test_input_x[i* batch_manager.batch_size:(i+1)* batch_manager.batch_size]
                        test_input_y = batch_manager.test_input_y[i* batch_manager.batch_size:(i+1)* batch_manager.batch_size]
                        step_loss,accuracy,_ = model_obj.train_step(sess,test_input_x,test_input_y,False)
                        valid_loss += step_loss * batch_manager.batch_size
                        valid_accuracy += accuracy
                    valid_loss = valid_loss /(test_epoch * batch_manager.batch_size)
                    valid_accuracy = valid_accuracy / test_epoch
                    print("验证集上面的loss值为 %.2f,准确度为 %.2f,学习率%.2f, Preplexity值为 %.2f" %(valid_loss,valid_accuracy,model_obj.learning_rate.eval(),math.exp(valid_loss)))

                #保存模型
                if (model_obj.global_step.eval()+1) % bi_lstm_setting.checkpoint_every == 0:
                    print("保存模型。。。。。。。")
                    checkpoint_path = model_obj.model_save_path
                    model_obj.saver.save(sess,checkpoint_path,global_step=model_obj.global_step)


if __name__ == '__main__':
    train()