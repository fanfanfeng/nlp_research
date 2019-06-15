# create by fanfan on 2019/6/6 0006
from dialog_system.transform_QAbot.tf_models_new.modules import get_token_embeddings,positional_encoding,multihead_attention,ff,label_smoothing,noam_scheme
import tensorflow as tf
from tqdm import tqdm
class Transformer():
    def __init__(self,params,train):
        self.params = params
        self.train = train
        self.embeddings = get_token_embeddings(
            self.params.vocab_size,self.params.num_units,zero_pad=False
        )


    def encode(self,x,training=True):
        '''
                Returns
                memory: encoder outputs. (N, T1, hidden_units)
                '''
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            enc = tf.nn.embedding_lookup(self.embeddings,x) # (N, T1, hidden_units)
            enc *= self.params.num_units ** 0.5  # scale

            enc += positional_encoding(enc,self.params.max_seq_length)
            enc = tf.layers.dropout(enc,self.params.dropout_rate,training=training)

            ## block
            for i in range(self.params.num_blocks):
                with tf.variable_scope('num_block_{}'.format(i),reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(queries=enc,
                                              keys=enc,
                                              values=enc,
                                              num_heads=self.params.num_heads,
                                              dropout_rate =self.params.dropout_rate,
                                              training=training)

                    # feed forward
                    enc = ff(enc,num_units=[self.params.d_ff,self.params.num_units])

        memory = enc
        return memory

    def decode(self,decoder_input,memory,training=True):
        '''
               memory: encoder outputs. (N, T1, hidden_units)

               Returns
               logits: (N, T2, V). float32.
               y_hat: (N, T2). int32
               y: (N, T2). int32
               '''
        with tf.variable_scope("decoder",reuse=tf.AUTO_REUSE):
            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings,decoder_input)
            dec *= self.params.num_units ** 0.5  # scale

            dec += positional_encoding(dec,self.params.max_seq_length)
            dec = tf.layers.dropout(dec,self.params.dropout_rate,training=training)

            # block
            for i in range(self.params.num_blocks):
                with tf.variable_scope('num_blocks_{}'.format(i),reuse=tf.AUTO_REUSE):
                    dec = multihead_attention(queries=dec,
                                              keys=dec,
                                              values=dec,
                                              num_heads=self.params.num_heads,
                                              dropout_rate=self.params.dropout_rate,
                                              training=training,
                                              scope='self_attention')

                    # vanilla attention
                    dec = multihead_attention(queries=dec,
                                              keys=memory,
                                              values=memory,
                                              num_heads=self.params.num_heads,
                                              dropout_rate=self.params.dropout_rate,
                                              training=training,
                                              scope='vanilla_attention')
                    # feed forward
                    dec = ff(dec,num_units=[self.params.d_ff,self.params.num_units])

        # Final linear projection (embedding weights are shared)
        weights = tf.transpose(self.embeddings) # (hidden_units, vocab_size)
        logits = tf.einsum('ntd,dk->ntk',dec,weights) #(N,T2,vocab_size)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits,y_hat

    def make_train(self,x,decoder_input,decoder_output):
        '''
                Returns
                loss: scalar.
                train_op: training operation
                global_step: scalar.
                summaries: training summary node
                '''
        # forward
        memory = self.encode(x)
        logits,preds = self.decode(decoder_input,memory)

        y_  = label_smoothing(tf.one_hot(decoder_output,depth=self.params.vocab_size))
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y_)

        nonpadding = tf.to_float(tf.not_equal(decoder_output,0)) # padding id
        loss = tf.reduce_sum(loss * nonpadding) /(tf.reduce_sum(nonpadding) + 1e-7)

        global_step = tf.train.get_or_create_global_step()
        learning_rate = noam_scheme(self.params.lr,global_step,self.params.warmup_steps)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss,global_step=global_step)

        tf.summary.scalar('lr',learning_rate)
        tf.summary.scalar('loss',loss)
        tf.summary.scalar('global_step',global_step)

        summaries = tf.summary.merge_all()
        return loss,train_op,global_step,summaries


    def create_eval(self,input):
        '''Predicts autoregressively
                At inference, input ys is ignored.
                Returns
                y_hat: (N, T2)
                '''
        memory, sents1 = self.encode(input, False)
        decoder_inputs = tf.ones((tf.shape(input)[0], 1), tf.int32) * 1 # start ID

        for _ in tqdm(range(self.params.max_seq_length)):
            logits,y_hat = self.decode(decoder_inputs,memory,False)
            _decoder_inputs = tf.concat((decoder_inputs,y_hat),1)
            decoder_inputs = _decoder_inputs

        return y_hat










