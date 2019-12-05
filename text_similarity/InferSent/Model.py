# create by fanfan on 2019/12/4 0004
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell,DropoutWrapper
from text_similarity.InferSent.Utils import print_shape

class InferSent(object):
    def __init__(self,seq_length,n_vocab,embedding_size,hidden_size,attention_size,n_classes,batch_size,learning_rate,optimizer,l2,clip_value):
        # model init
        self._parameter_init(seq_length,n_vocab,embedding_size,hidden_size,attention_size,n_classes,batch_size,learning_rate,optimizer,l2,clip_value)
        self._placeholder_init()

        # model operation
        self.logits = self._logits_op()
        self.loss = self._loss_op()
        self.acc = self._acc_op()
        self.train = self._training_op()

        tf.add_to_collection('train_mini', self.train)

    def _parameter_init(self,seq_length,n_vocab,embedding_size,hidden_size,attention_size,n_classes,batch_size,learning_rate,optimizer,l2,clip_value):
        self.seq_length = seq_length
        self.n_vocab = n_vocab
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.attention_size = attention_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.l2 = l2
        self.clip_value = clip_value

    def _placeholder_init(self):
        self.premise = tf.placeholder(tf.int32,[None,self.seq_length],'premise')
        self.hypothesis = tf.placeholder(tf.int32,[None,self.seq_length],'hypothesis')
        self.y = tf.placeholder(tf.float32,[None,self.n_classes],'y_true')
        self.premise_mask = tf.placeholder(tf.int32,[None],'premise_actual_length')
        self.hypothesis_mask = tf.placeholder(tf.int32,[None],"hypothesis_actual_length")
        self.embed_matrix = tf.placeholder(tf.float32,[self.n_vocab,self.embedding_size],'embedd_matrix')
        self.dropout_keep_prob = tf.placeholder(tf.float32,name='dropout_keep_prob')

    def _logits_op(self):
        u,v = self._biLSTMMaxEncodingBlock('biLSTM_Max_encoding')
        logits = self._compositionBlock(u,v,self.hidden_size,'composition')
        return logits



    def _biLSTMMaxEncodingBlock(self,scope):
        with tf.device('/cpu:0'):
            self.Embedding = tf.get_variable("Embedding",[self.n_vocab,self.embedding_size],tf.float32)
            self.embedding_left = tf.nn.embedding_lookup(self.Embedding,self.premise)
            self.embedding_right = tf.nn.embedding_lookup(self.Embedding,self.hypothesis)
            print_shape('embeded_left',self.embedding_left)
            print_shape('embeded_right',self.embedding_right)

        with tf.variable_scope(scope):
            outputsPremise,finalStatePremise = self._biLSTMBlock(self.embedding_left,self.hidden_size,'biLSTM',self.premise_mask)
            outputsHypothesis,finalStateHypothesis = self._biLSTMBlock(self.embedding_right,self.hidden_size,'biLSTM',self.hypothesis_mask,isReuse=True)

            u_premise = tf.concat(outputsPremise,axis=2)
            v_hypothesis = tf.concat(outputsHypothesis,axis=2)
            print_shape('u_premise',u_premise)
            print_shape('v_hypothesis',v_hypothesis)

            u = tf.reduce_max(u_premise,axis=1)
            v = tf.reduce_max(v_hypothesis,axis=1)
            print_shape('u',u)
            print_shape('v',v)
            return u,v





    def _biLSTMBlock(self,inputs,num_units,scope,seq_len=None,isReuse=False):
        with tf.variable_scope(scope,reuse=isReuse):
            lstmCell = LSTMCell(num_units = num_units)
            dropLSTMCell = lambda :DropoutWrapper(lstmCell,output_keep_prob=self.dropout_keep_prob)
            fwLSTMCell ,bwLSTMCell = dropLSTMCell(),dropLSTMCell()
            output = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = fwLSTMCell,
                cell_bw = bwLSTMCell,
                inputs = inputs,
                sequence_length = seq_len,
                dtype = tf.float32
            )
            return output

    # compositon block
    def _compositionBlock(self,u,v,hiddenSize,scope):
        with tf.variable_scope(scope):
            diff = tf.abs(tf.subtract(u,v))
            mul = tf.multiply(u,v)
            print_shape('diff',diff)
            print_shape('mul',mul)

            features = tf.concat([u,v,diff,mul],axis=1)
            print('features',features)

            y_hat = self._feedForwardBlock(features,self.hidden_size,self.n_classes,'feed_forward')
            print_shape('y_hat',y_hat)
            return y_hat


    def _feedForwardBlock(self,inputs,hidden_dims,num_units,scope,isReuse=False,initializer =None):
        with tf.variable_scope(scope,reuse=isReuse):
            if initializer is None:
                initializer = tf.random_normal_initializer(0.0,.1)

            with tf.variable_scope('feed_forward_layerl'):
                inputs = tf.nn.dropout(inputs,self.dropout_keep_prob)
                outputs = tf.layers.dense(inputs,hidden_dims,tf.nn.relu,kernel_initializer=initializer)

            with tf.variable_scope('feed_forward_layer2'):
                outputs = tf.nn.dropout(outputs,self.dropout_keep_prob)
                results = tf.layers.dense(outputs,num_units,tf.nn.tanh,kernel_initializer=initializer)
                return results

    def _loss_op(self,l2_lambda=0.0001):
        with tf.name_scope('cost'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y,logits=self.logits)
            loss = tf.reduce_mean(losses,name='loss_val')
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            loss += l2_loss
        return loss

    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = tf.argmax(self.logits,1,name='label_pred')
            label_true = tf.argmax(self.y,1,name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred,tf.int32),tf.cast(label_true,tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name='Accuracy')
        return accuracy


    def _training_op(self):
        with tf.name_scope("training"):
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate,momentum=0.9)
            elif self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)


        gradients,v = zip(*optimizer.compute_gradients(self.loss))
        if self.clip_value is not None:
            gradients,_ = tf.clip_by_global_norm(gradients,self.clip_value)

        train_op = optimizer.apply_gradients(zip(gradients,v))
        return train_op

