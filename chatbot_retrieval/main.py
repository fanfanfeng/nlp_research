# create by fanfan on 2018/4/13 0013

# 第一步加载训练书籍
import sys
sys.path.append(r'/data/python_project/nlp_research')
from DualLSTMEncoderRankModel.ranker import Ranker
from DualLSTMEncoderRankModel import config
from DualLSTMEncoderRankModel.train_input import BatchManager
import datetime
from tqdm import tqdm

import tensorflow as tf

def restore_model(sess,saver):
    ckpt = tf.train.get_checkpoint_state(config.model_save_path)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("restore model from {}".format(ckpt.model_checkpoint_path))
        saver.restore(sess,ckpt.model_checkpoint_path)



def mainTrain(sess):
    batch_manager = BatchManager(config.corpus_to_id_path)
    #batches_valid = batch_manager.get_valid_batches()
    #batches_test = batch_manager.get_test_batches()

    global_step = 0
    for epoch in range(config.numEpochs):
        print("----- Epoch {}/{} ; (lr={}) -----".format(epoch + 1, config.numEpochs, config.learning_rate))

        train_batches = batch_manager.get_training_batches()
        start_time = datetime.datetime.now()
        for nextBatch in tqdm(train_batches, desc="Training"):
            ops,feedDict = model_train.step(nextBatch)
            assert len(ops) == 3
            _,loss,train_summaries = sess.run(ops,feedDict)

            global_step += 1
            if global_step % 100 == 0:
                tqdm.write("----- Step %d -- CE Loss %.2f" % (global_step, loss))

            if global_step % config.saveEvery == 0 :
                train_writer.add_summary(train_summaries,global_step)
                train_writer.flush()

                print("开始在 valid_data上面测试")
                valid_losses = [0,0,0]
                for nextEvalBatch in batches_valid:
                    ops,feedDict = model_valid.step(nextEvalBatch)
                    assert  len(ops) == 2
                    loss,eval_summaries = sess.run(ops,feedDict)
                    for i in range(3):
                        valid_losses[i] += loss[i]






mode = "train"
graph = tf.Graph()
with tf.device('/gpu:0'):
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    ))

    with tf.name_scope("training"):
        model_train = Ranker(mode=tf.contrib.learn.ModeKeys.TRAIN)
    sess.run(tf.global_variables_initializer())
    graph_info = sess.graph
    train_writer = tf.summary.FileWriter(config.tf_log_train_path, graph_info)
    valid_writer = tf.summary.FileWriter(config.tf_log_valid_path, graph_info)
    ckpt_model_saver = tf.train.Saver(name='checkpoint_model_saver')
    restore_model(sess, ckpt_model_saver)

    if mode == "train":
        mainTrain(sess)