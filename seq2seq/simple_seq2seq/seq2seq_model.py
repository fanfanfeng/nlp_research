# create by fanfan on 2017/8/25 0025
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq
from tensorflow.contrib import rnn

from src.chatbot.simple_seq2seq import config
from src.chatbot.simple_seq2seq import data_utils


class Seq2SeqModel(object):
    """Sequence-to-sequence model with attention and for multiple buckets.
    
    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
    or into the seq2seq library for complete model implementation.
    This class also allows to use GRU cells in addition to LSTM cells, and
    sampled softmax to handle large output vocabulary size. A single-layer
    version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
    and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
    """
    def __init__(self, use_lstm=False,num_samples=512, forward_only=False):
        self.source_vocab_size = config.vocabulary_size
        self.target_vocab_size = config.vocabulary_size
        self.buckets = config.BUCKETS
        self.batch_size = config.FLAGS.batch_size
        self.learning_rate = tf.Variable(float(config.FLAGS.learning_rate),trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * config.FLAGS.learning_rate_decay_factor
        )
        self.lsmt_size = config.FLAGS.lstm_size
        self.num_layers = config.FLAGS.num_layers
        self.dropout = config.FLAGS.dropout
        self.max_gradient_norm = config.FLAGS.max_gradient_norm
        self.global_step = tf.Variable(0,trainable=False)
        self.model_dir = config.model_dir



        # If we use sampled softmax, we need an output projection.
        output_projection = None
        softmax_loss_function = None

        # Sampled softmax only makes sense if we sample less than vocabulary size.
        if num_samples > 0 and num_samples < self.target_vocab_size:
            w = tf.get_variable('proj_w',[self.lsmt_size,self.target_vocab_size])
            w_t = tf.transpose(w)
            b = tf.get_variable('proj_b',[self.target_vocab_size])
            output_projection = (w,b)

            def sampled_loss(labels,logits):
                labels = tf.reshape(labels, [-1, 1])  # Add one dimension (nb of true classes, here 1)

                # We need to compute the sampled_softmax_loss using 32bit floats to
                # avoid numerical instabilities.
                localWt = tf.cast(w_t, tf.float32)
                localB = tf.cast(b, tf.float32)
                localInputs = tf.cast(logits, tf.float32)

                return tf.cast(
                    tf.nn.sampled_softmax_loss(
                        localWt,  # Should have shape [num_classes, dim]
                        localB,
                        labels,
                        localInputs,
                        num_samples,  # The number of classes to randomly sample per batch
                        self.target_vocab_size),  # The number of classes
                    tf.float32)

            softmax_loss_function = sampled_loss


        # Create the internal multi-layer cell for our RNN.
        single_call = rnn.GRUCell(self.lsmt_size)
        if use_lstm:
            single_call = rnn.BasicLSTMCell(self.lsmt_size)

        if not forward_only:
            single_call = rnn.DropoutWrapper(single_call,input_keep_prob=1.0,output_keep_prob=self.dropout)

        cell = single_call
        if self.num_layers > 1:
            cell = rnn.MultiRNNCell([single_call] * self.num_layers)


        # The seq2seq function: we use embedding for the input and attention.
        def seq2seq_f(encoder_inputs,decoder_inputs,do_decode):
            import copy
            temp_cell = copy.deepcopy(cell)
            return legacy_seq2seq.embedding_attention_seq2seq(encoder_inputs,decoder_inputs,temp_cell,
                                                              num_encoder_symbols = self.source_vocab_size,
                                                              num_decoder_symbols = self.target_vocab_size,
                                                              embedding_size= self.lsmt_size,
                                                              output_projection = output_projection,
                                                              feed_previous = do_decode)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(self.buckets[-1][0]):
            self.encoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name="encoder{0}".format(i)))

        for i in range(self.buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.int32,shape=[None],name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i+1] for i in range(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only:
            self.outputs,self.losses = legacy_seq2seq.model_with_buckets(
                self.encoder_inputs,self.decoder_inputs,targets,
                self.target_weights,self.buckets,lambda x,y:seq2seq_f(x,y,True),
                softmax_loss_function = softmax_loss_function
            )

            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in range(len(self.buckets)):
                    self.outputs[b] = [
                        tf.matmul(output,output_projection[0]) + output_projection[1] for output in self.outputs[b]
                    ]
        else:
            self.outputs, self.losses = legacy_seq2seq.model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, self.buckets,
                lambda x, y: seq2seq_f(x, y, False),
                softmax_loss_function=softmax_loss_function)


        # Gradients and SGD update operation for training the model.
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            params = tf.trainable_variables()
            for b in range(len(self.buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,self.max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))


        self.saver = tf.train.Saver(tf.all_variables(),max_to_keep=3)
        self.mergedSummaries = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(config.graph_dir)


    def step(self,session,encoder_inputs,decoder_inputs,target_weights,bucket_id,forward_only):
        """Run a step of the model feeding the given inputs.

            Args:
              session: tensorflow session to use.
              encoder_inputs: list of numpy int vectors to feed as encoder inputs.
              decoder_inputs: list of numpy int vectors to feed as decoder inputs.
              target_weights: list of numpy float vectors to feed as target weights.
              bucket_id: which bucket of the model to use.
              forward_only: whether to do the backward step or only forward.

            Returns:
              A triple consisting of gradient norm (or None if we did not do backward),
              average perplexity, and the outputs.

            Raises:
              ValueError: if length of encoder_inputs, decoder_inputs, or
                target_weights disagrees with bucket size for the specified bucket_id.
            """
        # Check if the sizes match.
        encoder_size,decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket, %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,%d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,%d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for id in range(encoder_size):
            input_feed[self.encoder_inputs[id].name] = encoder_inputs[id]
        for id in range(decoder_size):
            input_feed[self.decoder_inputs[id].name] = decoder_inputs[id]
            input_feed[self.target_weights[id].name] = target_weights[id]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size],dtype=np.int32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],
                           self.gradient_norms[bucket_id],
                           self.losses[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            for l in range(decoder_size):
                output_feed.append(self.outputs[bucket_id][l])


        outputs = session.run(output_feed,input_feed)
        if not forward_only:
            return outputs[1],outputs[2],None
        else:
            return  None,outputs[0],outputs[1:]


    def get_batch(self,data,bucket_id):
        """Get a random batch of data-en from the specified bucket, prepare for step.

            To feed data-en in step(..) it must be a list of batch-major vectors, while
            data-en here contains single length-major cases. So the main logic of this
            function is to re-index data-en cases to be in the proper format for feeding.

            Args:
              data-en: a tuple of size len(self.buckets) in which each element contains
                lists of pairs of input and output data-en that we use to create a batch.
              bucket_id: integer, which bucket to get the batch for.

            Returns:
              The triple (encoder_inputs, decoder_inputs, target_weights) for
              the constructed batch that has the proper format to call step(...) later.
            """
        encoder_size,decoder_size = self.buckets[bucket_id]
        encoder_inputs,decoder_inputs = [],[]

        # Get a random batch of encoder and decoder inputs from data-en,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input,decoder_input = random.choice(data[bucket_id])

            # Encoder inputs are padded and then reversed.
            encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
            encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - len(decoder_input) - 1
            decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)

        # Now we create batch-major vectors from the data-en selected above.
        batch_encoder_inputs,batch_decoder_inputs,batch_weights = [], [],[]

         # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)],dtype=np.int32)
            )

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx]
                          for batch_idx in range(self.batch_size)],dtype=np.int32)
            )

            # Create target_weights to be 0 for targets that are padding
            batch_weight = np.ones(self.batch_size,dtype=np.float32)
            for batch_idx in range(self.batch_size):
                # We set weight to 0 if the corresponding target is a PAD symbol.
                # The corresponding target is decoder_input shifted by 1 forward.
                if length_idx < decoder_size - 1:
                    target = decoder_inputs[batch_idx][length_idx + 1]
                if length_idx == decoder_size -1 or target == data_utils.PAD_ID:
                    batch_weight[batch_idx] = 0.0
            batch_weights.append(batch_weight)

        return batch_encoder_inputs,batch_decoder_inputs,batch_weights

    @staticmethod
    def model_create_or_restore(session,forward_only):
        """Create translation model and initialize or load parameters in session."""
        model = Seq2SeqModel(forward_only = forward_only )
        ckpt = tf.train.get_checkpoint_state(model.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("从 %s 中恢复模型" % ckpt.model_checkpoint_path)
            model.saver.restore(session,ckpt.model_checkpoint_path)
        else:
            print("创建新的模型")
            session.run(tf.global_variables_initializer())
        return model

    @staticmethod
    def get_predicted_sentence(input_sentence,vocab,rev_vocab,model,sess):
        input_token_ids = data_utils.sentence_to_token_ids(input_sentence,vocab)

        # Which bucket does it belong to?
        bucket_id = min([b for b in range(len(model.buckets)) if model.buckets[b][0] > len(input_token_ids)])
        outputs = []

        feed_data = { bucket_id: [(input_token_ids,outputs)]}
        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs,decoder_inputs,target_weights = model.get_batch(feed_data,bucket_id)

        # Get output logits for the sentence.
        _,_,output_logits = model.step(sess,encoder_inputs,decoder_inputs,target_weights,bucket_id,forward_only=True)

        outputs = []
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        for logit in output_logits:
            selected_token_id = int(np.argmax(logit,axis=1))
            if selected_token_id == data_utils.EOS_ID:
                break
            else:
                outputs.append(selected_token_id)

        # Forming output sentence on natural language
        output_sentence = " ".join([rev_vocab[output] for output in outputs])
        return  output_sentence









