# create by fanfan on 2017/8/26 0026
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
import data_utils
import config
import model
import numpy as np
import time
import math



def train():
    tran_batch_manager = data_utils.BatchManager("data/train.txt.id40000.in", config.batch_size)
    test_batch_manager = data_utils.BatchManager("data/test.txt.id40000.in", config.batch_size)

    with tf.Session() as sess:
        graph_writer = tf.summary.FileWriter(config.model_dir,graph=sess.graph)
        model_obj = model.Seq2SeqModel('train')
        model_obj.model_restore(sess)


        #outputTensors = []
        #print(model_obj.decoder_pred_decode.name.replace(":0",""))
        #outputTensors.append(model_obj.decoder_pred_decode.name.replace(":0",""))

        #output_graph_with_weight = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def,outputTensors)
        #with tf.gfile.FastGFile(os.path.join(config.model_dir, "weight_seq2seq.pb"),
        #                        'wb') as gf:
        #    gf.write(output_graph_with_weight.SerializeToString())

        print("开始训练")
        loss = 0.0
        start_time = time.time()
        best_loss = 10000.0
        for epoch_id  in range(config.max_epochs):
            for step, train_batch in enumerate(tran_batch_manager.iterbatch()):
                if train_batch['encode'] is None:
                    continue
                print("开始第%d轮第%d次训练,globa_step %d" % (epoch_id+1,step+1,model_obj.global_step.eval()))

                # Execute a single training step
                step_loss,summary = model_obj.train(sess,encoder_inputs=train_batch['encode'],
                                                decoder_inputs=train_batch['decode'],
                                                encoder_inputs_length=train_batch['encode_lengths'],
                                                decoder_inputs_length=train_batch['decode_lengths'])
                loss += float(step_loss) / config.display_freq

                if (model_obj.global_step.eval() +1) % config.display_freq == 0:
                    if loss < best_loss:
                        best_loss = loss
                        print("保存模型。。。。。。。")
                        checkpoint_path = model_obj.mode_save_path
                        model_obj.saver.save(sess, checkpoint_path, global_step=model_obj.global_step)

                    avg_perplexity = math.exp(float(loss)) if loss <300 else float("inf")

                    #计算时间
                    time_cost = time.time() - start_time
                    step_time = time_cost / config.display_freq
                    print('第%d轮训练，第%d的步，loss值为 %.2f , Preplexity值为 %.2f,花费时间 %f' % (epoch_id, model_obj.global_step.eval(),loss,avg_perplexity,step_time))
                    loss = 0.0
                    start_time = time.time()

                    # Record training summary for the current batch
                    graph_writer.add_summary(summary, model_obj.global_step.eval())

                #验证模型
                if (model_obj.global_step.eval() +1) % config.valid_freq == 0:
                    print("验证模型。。。。。")
                    valid_loss = 0.0
                    totoal_sentent = 0
                    for test_batch in test_batch_manager.iterbatch():
                        step_loss,summary = model_obj.eval(sess,encoder_inputs=test_batch['encode'],
                                                decoder_inputs=test_batch['decode'],
                                                encoder_inputs_length=test_batch['encode_lengths'],
                                                decoder_inputs_length=test_batch['decode_lengths'])
                        batch_size = test_batch['encode_lengths'].shape[0]
                        valid_loss += step_loss * batch_size
                        totoal_sentent += batch_size
                    valid_loss = valid_loss /totoal_sentent
                    print("验证集上面的loss值为 %.2f, Preplexity值为 %.2f" %(valid_loss,math.exp(valid_loss)))


                if (model_obj.global_step.eval()+1) % config.save_freq == 0:
                    print("保存模型。。。。。。。")
                    checkpoint_path = model_obj.mode_save_path
                    model_obj.saver.save(sess,checkpoint_path,global_step=model_obj.global_step)


if __name__ == '__main__':
    train()