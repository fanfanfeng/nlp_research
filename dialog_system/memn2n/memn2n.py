# create by fanfan on 2018/8/29 0029
import tensorflow as tf
import numpy as np

def position_encoding(sentence_size,embedding_size):
    '''
    词袋模型本身是无序的，句子“我爱你”和“你爱我”在BOW中都是{我，爱，你}，模型本无法区分这两句话不同的含义，但如果给每个词加上position encoding，变成{我1，爱2， 你3}和{我3，爱2，你1}，则变成不同的数据，所以就是位置编码就是一种特征。一般用于：1.非RNN网络，由于不能编码序列顺序。需要显示的输入位置信息。如CNN seq2seq 2.需要加强位置信息。如句子中存在核心词，需要对核心词的周边词着重加权等。。
    :param sentence_size: 
    :param embedding_size: 
    :return: 
    '''
    encoding = np.ones((embedding_size,sentence_size),dtype=np.float32)
    ls =  sentence_size + 1
    le = embedding_size + 1
    for i in range(1,le):
        for j in range(1,ls):
            encoding[i-1,j-1] = (i - (embedding_size + 1)/2) * ( j - (sentence_size + 1)/2)
    encoding = 1 + 4 * encoding/embedding_size/sentence_size

    # Make position encoding of time words identity to avoid modifying them
    encoding[:,-1] = 1.0
    return np.transpose(encoding)


def add_gradient_noise(t,stddev=1e-3,name=None):
    with tf.name_scope('add_gradient_noise') as name:
        t = tf.convert_to_tensor(t,name='t')
        gn = tf.random_normal(tf.shape(t),stddev=stddev)
        return tf.add(t,gn,name=name)


class MemN2N(object):
    def __init__(self,batch_size,vocab_size,sentence_size,memory_size,embedding_size,hops=3,
                 max_grad_norm=40,
                 initializer = tf.random_normal_initializer(stddev=0.1),
                 session = tf.Session(),
                 name='MemN2N'):
        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._hops = hops
        self._max_grad_norm = max_grad_norm
        self._initializer = initializer
        self._name = name
        self._embedding_size = embedding_size

        self.build_inputs()

        self._opt = tf.train.GradientDescentOptimizer(learning_rate=self._learnrate)

        self._position_encoding = position_encoding(self._sentence_size,self._embedding_size)#tf.constant(position_encoding(self._sentence_size,self._embedding_size),name='position_encoding')
        logits = self._inference(self._stores,self._queries)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=tf.cast(self._answers,tf.float32),name='cross_entropy')
        cross_entropy_sum = tf.reduce_sum(cross_entropy,name='cross_entropy_sum')

        loss_op = cross_entropy_sum

        grads_and_vars = self._opt.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g,self._max_grad_norm),v) for g,v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g),v) for g,v in grads_and_vars]
        train_op = self._opt.apply_gradients(grads_and_vars,name='train_op')

        predict_op = tf.argmax(logits,1,name='predict_op')
        predict_proba_op = tf.nn.softmax(logits,name='predict_proba_op')
        predict_log_proba_op = tf.log(predict_proba_op,name='predict_log_proba_op')

        # assign ops
        self.loss_op = loss_op
        self.predict_op = predict_op
        self.predict_proba_op = predict_proba_op
        self.predict_log_proba_op = predict_log_proba_op
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)






    def build_inputs(self):
        self._stores = tf.placeholder(tf.int32,[None,self._memory_size,self._sentence_size],name='stories')
        self._queries = tf.placeholder(tf.int32,[None,self._sentence_size],name='queries')
        self._answers = tf.placeholder(tf.int32,[None,self._vocab_size],name='answers')
        self._learnrate = tf.placeholder(tf.float32,[],name='learning_rate')

        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1,self._embedding_size])
            A = tf.concat(axis=0,values=[nil_word_slot,self._initializer([self._vocab_size -1,self._embedding_size])])
            C = tf.concat(axis=0,values=[nil_word_slot,self._initializer([self._vocab_size -1,self._embedding_size])])

            self.A_1 = tf.Variable(A,name='A')
            self.C = []
            for hopn in range(self._hops):
                with tf.variable_scope('hop_{}'.format(hopn)):
                    self.C.append(tf.Variable(C, name="C"))


            self._nil_vars = set([self.A_1.name] + [x.name for x in self.C])
            j=0

    def _inference(self,stories,queries):
        with tf.variable_scope(self._name):
            q_emb = tf.nn.embedding_lookup(self.A_1,queries)
            u_0 = tf.reduce_sum(q_emb * self._position_encoding,1)
            u = [u_0]

            for hopn in range(self._hops):
                if hopn == 0 :
                    m_emb_A = tf.nn.embedding_lookup(self.A_1,stories)
                    m_A = tf.reduce_sum(m_emb_A * self._position_encoding,2)
                else:
                    with tf.variable_scope('hop_{}'.format(hopn - 1)):
                        m_emb_A = tf.nn.embedding_lookup(self.A_1,stories)
                        m_A = tf.reduce_mean(m_emb_A * self._position_encoding,2)

                u_temp = tf.transpose(tf.expand_dims(u[-1],-1),[0,2,1])
                dotted = tf.reduce_sum(m_A * u_temp ,2)

                probs = tf.nn.softmax(dotted)
                probs_temp = tf.transpose(tf.expand_dims(probs,-1),[0,2,1])

                with tf.variable_scope("hop_{}".format(hopn)):
                    m_emb_C = tf.nn.embedding_lookup(self.C[hopn],stories)
                m_C  = tf.reduce_sum(m_emb_C * self._position_encoding,2)
                c_temp = tf.transpose(m_C,[0,2,1])
                o_k = tf.reduce_sum(c_temp * probs_temp,2)

                u_k = u[-1] + o_k

                u.append(u_k)

            with tf.variable_scope('hop_{}'.format(self._hops)):
                return tf.matmul(u[-1],tf.transpose(self.C[-1],[1,0]))



    def batch_fit(self,stories,queries,answers,learning_rate):
        feed_dict = {self._stores:stories,self._queries:queries,self._answers:answers,self._learnrate:learning_rate}
        loss,_ = self._sess.run([self.loss_op,self.train_op],feed_dict=feed_dict)
        return loss

    def predict(self,stories,queries):
        feed_dict = {self._stores:stories,self._queries:queries}
        return self._sess.run(self.predict_op,feed_dict=feed_dict)


    def predict_proba(self, stories, queries):
        """Predicts probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)

        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stores: stories, self._queries: queries}
        return self._sess.run(self.predict_proba_op, feed_dict=feed_dict)

    def predict_log_proba(self, stories, queries):
        """Predicts log probabilities of answers.

        Args:
            stories: Tensor (None, memory_size, sentence_size)
            queries: Tensor (None, sentence_size)
        Returns:
            answers: Tensor (None, vocab_size)
        """
        feed_dict = {self._stores: stories, self._queries: queries}
        return self._sess.run(self.predict_log_proba_op, feed_dict=feed_dict)








if __name__ == '__main__':
    MemN2N(batch_size=24,
           vocab_size=25,
           sentence_size=25,
           memory_size=25,
           embedding_size=25,
           )