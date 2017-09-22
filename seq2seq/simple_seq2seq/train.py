# create by fanfan on 2017/8/26 0026
import tensorflow as tf
from src.chatbot.simple_seq2seq import data_utils
from src.chatbot.simple_seq2seq import config
from src.chatbot.simple_seq2seq import seq2seq_model
import numpy as np
import time
import os

def train():
    print("从文件夹%s中准备语料" % config.data_dir)
    train_data,dev_data,_ = data_utils.prepare_data_for_model()


    #读取数据并载入内存
    dev_set = data_utils.read_data(dev_data)
    train_set = data_utils.read_data(train_data)

    train_bucked_sizes = [ len(train_set[b]) for b in range(len(config.BUCKETS))]
    train_total_size = float(sum(train_bucked_sizes))

    train_buckeds_scale = [ sum(train_bucked_sizes[:i+1]) /train_total_size for i in range(len(train_bucked_sizes))]

    #开始训练
    with tf.Session() as sess:
        # Create model.
        print("创建一个 %s 层 %d单元的模型" % (config.FLAGS.num_layers,config.FLAGS.lstm_size))
        model =  seq2seq_model.Seq2SeqModel.model_create_or_restore(sess,forward_only=False)
        model.writer.add_graph(sess.graph)

        step_time,loss = 0.,0.
        current_step = 0
        previous_losses = []
        while True:
            random_number_01 = np.random.random_sample()
            bucked_id = min([i for i in range(len(train_buckeds_scale)) if train_buckeds_scale[i] > random_number_01])

            # Get a batch and make a step
            start_time = time.time()
            encoder_inputs,decoder_inputs,target_weights = model.get_batch(train_set,bucked_id)

            _, step_loss,_ = model.step(sess,encoder_inputs,decoder_inputs,target_weights,bucked_id,forward_only=False)

            step_time += (time.time() - start_time) / config.FLAGS.steps_per_checkpoint
            loss += step_loss / config.FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % config.FLAGS.steps_per_checkpoint == 0:
                perplexity = np.math.exp(loss) if loss < 300 else float('inf')
                print("训练步数 %d, 学习率 %.4f ，花费时间 %.2f， perplexity %.2f ，loss %.2f" %
                      (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity,loss))

                # 如果loss3次以内都没有下降，则缩小学习率.
                if len(previous_losses) >= 2 and loss > max(previous_losses[-2:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)

                #保存模型
                checkpoint_path = os.path.join(model.model_dir,'model.ckpt')
                model.saver.save(sess,checkpoint_path,global_step= model.global_step)

                step_time,loss = 0.,0.

                #在测试数据上面评测
                for bucket_id in range(len(model.buckets)):
                    encoder_inputs,decoder_inputs,target_weights = model.get_batch(dev_set, bucket_id)
                    _,eval_loss,_ = model.step(sess,encoder_inputs,decoder_inputs,target_weights,bucket_id,forward_only=True)

                    eval_ppx = np.math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d perplexity %.2f ,loss %.2f" % (bucket_id, eval_ppx,eval_loss))

if __name__ == '__main__':
    train()