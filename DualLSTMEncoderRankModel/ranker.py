# create by fanfan on 2018/4/16 0016
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from DualLSTMEncoderRankModel import config
import os
class Ranker:
    def __init__(self,model="train"):
        '''
        model:["train", "valid", "eval", "predict", "cache"]
        :param model: 
        '''
        self.model = model
        self.dtype = tf.float32


        self.buildNetwork()

    def _create_rnn_cell(self):
        cell = rnn.BasicLSTMCell(config.hiddenSize)
        cell = rnn.DropoutWrapper(cell,output_keep_prob=config.dropout)
        return cell

    def buildNetwork(self):
        # 假设 batchsize = 100 ,句子长度20
        with tf.name_scope('placeholder_query'):
            # shape = [100,20]
            self.query_seqs = tf.placeholder(tf.int32,[None,None],name='query')
            # shape = [100]
            self.query_length = tf.placeholder(tf.int32,[None],name='query_length')

        with tf.name_scope('placeholder_labels'):
            # shape = [100,100]
            self.labels = tf.placeholder(tf.int32,[None,None],name='labels')
            # shape =
            self.targets = tf.placeholder(tf.int32,[None],name='targets')

        with tf.name_scope('placeholder_response'):
            # shape = [100,20]
            self.response_seqs = tf.placeholder(tf.int32,[None,None],name='response')
            # shape = [100]
            self.response_length = tf.placeholder(tf.int32,[None],name='response_length')

        with tf.name_scope('embedding_layer'):
            # shape=[40000,200]
            self.embedding = tf.get_variable('embedding',[config.max_vocab_size, config.embedding_size])
            # shape=[100,20,200]
            self.embed_query = tf.nn.embedding_lookup(self.embedding,self.query_seqs)
            # shape=[100,20,200]
            self.embed_response = tf.nn.embedding_lookup(self.embedding,self.response_seqs)

        encoder_cell = rnn.MultiRNNCell([self._create_rnn_cell() for _ in range(config.layer_num)])

        # shape = [100,20, 256] ,[h:(100,256),c:(100,256)]
        query_output,query_final_state = tf.nn.dynamic_rnn(
            cell= encoder_cell,
            inputs=self.embed_query,
            sequence_length=self.query_length,
            time_major=False,
            dtype=tf.float32
        )

        # shape = [100,20, 256] ,[100,256]
        response_output,response_final_state = tf.nn.dynamic_rnn(
            cell = encoder_cell,
            inputs=self.embed_response,
            sequence_length=self.response_length,
            time_major=False,
            dtype=tf.float32
        )

        with tf.variable_scope('bilinar_regression'):
            # shape= [256,256]
            W = tf.get_variable('bilinear_W',shape=[config.hiddenSize,config.hiddenSize],initializer=tf.truncated_normal_initializer())


        if self.model == 'train':
            # shape = [100,256]
            response_final_state = tf.matmul(response_final_state[-1].h,W)
            # shape = [100,100]
            logits = tf.matmul(
                a = query_final_state[-1].h,
                b = response_final_state,
                transpose_b=True
            )

            self.losses = tf.losses.softmax_cross_entropy(
                onehot_labels=self.labels,
                logits=logits
            )

            self.means_loss = tf.reduce_mean(self.losses,name='mean_loss')
            train_loss_summary = tf.summary.scalar('loss',self.means_loss)

            self.training_summaries = tf.summary.merge([train_loss_summary])

            opt = tf.train.AdamOptimizer(learning_rate= config.learning_rate,beta1=0.9,
                                         beta2=0.999,epsilon=1e-08)
            self.train_op = opt.minimize(self.means_loss)

        elif self.model=='valid' or self.model=='eval':
            # shape =[100,256]
            response_final_state = tf.matmul(response_final_state[-1].h,W)
            #[100* 30,256]
            query_final_state = tf.reshape(tf.tile(query_final_state[-1].h,[1,config.ranksize]),[-1,config.hiddenSize])

            total_score = tf.multiply(query_final_state,response_final_state)
            # shape =[ 3000,1]
            logits = tf.reduce_sum(
                total_score,
                axis=1,
                keep_dims=True
            )
            # shape = [100,30]
            logits = tf.reshape(logits,[-1,config.ranksize])

            self.response_top_1 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(
                    predictions=logits,
                    targets=self.targets,
                    k=1,
                    name='prediction_in_top_1'
                ),dtype=tf.float32)
            )

            self.response_top_3 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(
                    predictions=logits,
                    targets=self.targets,
                    k=3,
                    name= "prediction_in_top_3"
                ),dtype=tf.float32)
            )

            self.response_top_5 = tf.reduce_mean(
                tf.cast(tf.nn.in_top_k(
                        predictions= logits,
                        targets= self.targets,
                        k=5,
                        name='prediction_in_top_5'
                    )
                ,dtype=tf.float32)
            )

            response_top_1 = tf.summary.scalar('valid_top1_of%s' % config.ranksize,self.response_top_1)
            response_top_3 = tf.summary.scalar('valid_top3_of%s' % config.ranksize,self.response_top_3)
            response_top_5 = tf.summary.scalar('valid_top5_of%s' % config.ranksize,self.response_top_5)

            self.evaluation_summaries = tf.summary.merge(
                inputs=[response_top_1, response_top_3, response_top_5],
                name='eval_monitor')
            self.outputs = (self.response_top_1,self.response_top_3,self.response_top_5)

        elif self.model == "predict":
            cache_response_final_state, cache_response_seqs, cache_ranksize = self.load_cache_op()
            query_final_state =  tf.reshape(tf.tile(query_final_state[-1].h,[1,cache_ranksize]),[-1,config.hiddenSize])
            logits = tf.reduce_sum(
                tf.multiply(query_final_state,cache_response_final_state),
                axis=1,
                keep_dims=True
            )
            logits = tf.reshape(logits,[-1,cache_ranksize])
            logits = tf.nn.softmax(logits)
            self.outputs = logits
        elif self.model == 'cache':
            response_final_state = tf.matmul(response_final_state[-1].h, W)
            self.outputs = response_final_state

    def load_cache_op(self):
        g = tf.Graph()
        cache_sess = tf.Session(graph=g)
        with cache_sess.as_default():
            with g.as_default():
                cache_saver = tf.train.import_meta_graph(os.path.join(config.cache_path, 'cache.ckpt.meta'))
                ckpt_file = os.path.join(config.cache_path,'cache.ckpt')
                cache_saver.restore(cache_sess,ckpt_file)
                cache_response_final_state = cache_sess.run('cache_response_final_state:0')
                cache_response_seqs = cache_sess.run('cache_response_seqs:0')
                cache_ranksize = cache_response_final_state.shape[0]
        return (cache_response_final_state,cache_response_seqs,cache_ranksize)

    def step(self,batch):
        feedDict = {}
        ops = None

        if self.model != "cache":
            feedDict[self.query_seqs] = batch.query_seqs
            feedDict[self.query_length] = batch.query_length

        if self.model != 'predict':
            feedDict[self.response_seqs] = batch.response_seqs
            feedDict[self.response_length] = batch.response_length

        if self.model == 'train':
            ops = (self.train_op,self.means_loss,self.training_summaries)
            feedDict[self.labels] = np.eye(len(batch.query_seqs))
        elif self.model == 'valid' or self.model == 'eval':
            ops = (self.outputs,self.evaluation_summaries)
            feedDict[self.targets] = np.zeros((len(batch.query_seqs))).astype(int)
        elif self.model == 'cache':
            ops = self.outputs
        elif self.model == 'predict':
            ops = self.outputs
            feedDict[self.targets] = np.zeros((len(batch.query_seqs))).astype(int)
        return ops,feedDict






if __name__ == '__main__':
    Ranker(model='eval')
