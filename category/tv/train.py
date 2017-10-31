# create by fanfan on 2017/7/4 0004
import os
import sys
import numpy as np
import tensorflow as tf
from category.tv import classfication_setting
from category.tv import  bi_lstm_model
from category.tv import bi_lstm_model_attention
from category.tv import data_util



def train():


    graph = tf.Graph()
    with graph.as_default() as g,tf.Session(graph=g) as sess:
        if classfication_setting.use_attention:
            model = bi_lstm_model_attention.Bi_lstm()
            print("初始化attion模型完成")
        else:
            model = bi_lstm_model.Bi_lstm()
            print("初始化模型完成")


        graph_writer = tf.summary.FileWriter(classfication_setting.graph_model_bi_lstm, graph=sess.graph)
        tv_data = data_util.BatchManager(classfication_setting.tv_data_path, classfication_setting.batch_size)
        print("加载训练数据完成")
        check_point = tf.train.get_checkpoint_state(model.train_model_save_path)
        if check_point and tf.train.checkpoint_exists(check_point.model_checkpoint_path):
            print("reading model from %s" % check_point.model_checkpoint_path)
            model.saver.restore(sess, check_point.model_checkpoint_path)
        else:
            print("create model ")
            sess.run(tf.global_variables_initializer())
        #model.restore_model(sess)


        #保存模型graph
        #tf.train.write_graph(sess.graph_def, classfication_setting.graph_model_bi_lstm, "weight_classify.pb", False)

         #将权重固话到graph中去
        #output_tensor = []
        #output_tensor.append(model.logits.name.replace(":0",""))
        #output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, output_tensor)
        #with tf.gfile.FastGFile(os.path.join(classfication_setting.graph_model_bi_lstm, "weight_classify.pb"),
         #                      'wb') as gf:
         #   gf.write(output_graph_with_weight.SerializeToString())

        for num_epoch in range(classfication_setting.num_epochs):
            print("epoch  {}".format(num_epoch +1))
            for train_x,train_y in tv_data.train_iterbatch():
                step,_,_,_,_ = model.train_step(sess,is_train=True,inputX= train_x,inputY = train_y)
                if step % classfication_setting.show_every == 0:
                    _,learning_rate,loss,accuracy,summary_op = model.train_step(sess,is_train=True,inputX= train_x,inputY = train_y)
                    print("第{}轮训练, 训练步数 {}, 学习率 {:g}, 损失值 {:g}, 精确值 {:g}".format(num_epoch + 1, step, learning_rate, loss, accuracy))
                    graph_writer.add_summary(summary_op,step)
                if step % classfication_setting.valid_every == 0:
                    avg_loss = 0
                    avg_accuracy = 0
                    for test_x, test_y in tv_data.test_iterbatch():
                        _,_,loss,accuracy,_ = model.train_step(sess,False,test_x,test_y)
                        avg_loss += loss
                        avg_accuracy += accuracy

                    avg_loss = avg_loss / tv_data.test_epoch
                    avg_accuracy = avg_accuracy /tv_data.test_epoch
                    print("验证模型, 训练步数 {} ,学习率 {:g}, 损失值 {:g}, 精确值 {:g}".format(step, model.learning_rate.eval(), avg_loss, avg_accuracy))


                if step % classfication_setting.checkpoint_every == 0:
                    path = model.saver.save(sess,classfication_setting.train_model_bi_lstm,step)
                    print("模型保存到{}".format(path))



if __name__ == '__main__':
    train()





