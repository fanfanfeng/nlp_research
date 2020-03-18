# create by fanfan on 2020/3/17 0017
import tensorflow as tf
import third_models.albert_zh.modeling_google as modeling
import tensorflow_estimator as tf_estimator
import third_models.albert_zh.optimization as optimization
from Competition.ai_challenger_2018_sentiment_analysis import settings
from tensorflow.contrib.layers import fully_connected,conv1d
from utils.bert import tokenization
params = settings.ParamsModel()
tokenizer = tokenization.FullTokenizer(
        vocab_file=settings.bert_model_vocab_path, do_lower_case=True)
params.char2id = tokenizer.vocab

def get_setence_length(data, name):
    used = tf.sign(tf.abs(data))
    length = tf.reduce_sum(used, reduction_indices=-1)
    length = tf.cast(length, tf.int32, name=name)
    return length



def create_model(albert_config,is_training,input_ids,input_mask,segment_ids,
                 labels,aspechts_char):
    model = modeling.AlbertModel(
        config=albert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask= input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False
    )
    context_embedding = model.get_sequence_output()
    #contexts_ch_lens = get_setence_length(input_ids,
    #                                           'contexts_ch_lens')  # tf.placeholder(tf.int32, [None],name="contexts_ch_lens")
    #contexts_ch_lens = tf.reshape(contexts_ch_lens, [-1])

    aspects_ch = tf.expand_dims(
        tf.get_variable(initializer=aspechts_char, name='aspect_char', dtype=tf.int64, trainable=False), axis=0)

    with tf.variable_scope("aspect_layer"):
        embedding_matrix = tf.get_variable(name='embedding_ch',
                                                   shape=[len(params.char2id.keys()), params.embedding_dim],
                                                      trainable=True, dtype=tf.float32)
        aspects_ch = tf.reshape(aspects_ch,shape=[-1,params.max_char_len])
        aspect_inputs = tf.nn.embedding_lookup(embedding_matrix, aspects_ch)
        aspects_ch_lens = get_setence_length(aspects_ch, "aspects_ch_lens")
        aspects_ch_lens = tf.reshape(aspects_ch_lens, [-1])
        cell_fw = tf.contrib.rnn.GRUCell(params.hiden_sizes)
        cell_bw = tf.contrib.rnn.GRUCell(params.hiden_sizes)
        _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, aspect_inputs, aspects_ch_lens, dtype=tf.float32)
        aspect_emb = tf.concat([state_fw, state_bw], axis=1)
        aspect_emb = tf.reshape(aspect_emb,
                                [-1, params.n_class , 2 * params.hiden_sizes * params.max_aspect_len])

        new_aspects = fully_connected(aspect_emb, params.kernel_num, activation_fn=None)
        all_aspects = tf.split(new_aspects, [1] * params.n_class, 1)

    with tf.variable_scope('context_layer'):
        content_reps = conv1d(context_embedding, params.kernel_num, params.kernel_sizes)
        if is_training:
            content_reps = tf.nn.dropout(content_reps,keep_prob=params.dropout_keep)


    with tf.variable_scope('gate_cnn'):
        represent_reps = []
        for idx, a_aspect in enumerate(all_aspects):
            with tf.variable_scope("context_conv_" + str(idx), reuse=tf.AUTO_REUSE):
                aspect_rel_reps = conv1d(context_embedding, params.kernel_num, params.kernel_sizes)

            x = tf.multiply(tf.nn.relu(a_aspect + aspect_rel_reps), content_reps)

            with tf.variable_scope('represent_conv_' + str(idx)):
                repre = conv1d(x, params.kernel_num, params.kernel_sizes)

                max_pool_repre = tf.layers.max_pooling1d(repre, repre.get_shape().as_list()[1],
                                                         repre.get_shape().as_list()[1])
                repre_last = tf.squeeze(max_pool_repre, axis=1)

            with tf.variable_scope('full_connect_' + str(idx)):
                if is_training:
                    repre_last = tf.nn.dropout(repre_last, keep_prob=0.9)
                output_repre = fully_connected(repre_last, params.n_sub_class, activation_fn=None,
                                               weights_initializer=tf.random_uniform_initializer(-0.1, 0.1),
                                               weights_regularizer=tf.contrib.layers.l2_regularizer(params.l2_reg))
                represent_reps.append(output_repre)

    with tf.variable_scope("output_layer"):
        logit = tf.concat(represent_reps, 1)
        logit = tf.reshape(logit, [-1, params.n_class, params.n_sub_class])

        predictions = tf.argmax(logit, axis=-1, output_type=tf.int32)
        probabilities = tf.nn.softmax(logit, axis=-1)
        log_probs = tf.nn.log_softmax(logit, axis=-1)

    #with tf.variable_scope("loss"):
    per_example_loss = -tf.reduce_sum(tf.cast(labels,dtype=tf.float32) * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss,name='loss')

    return (loss,probabilities,predictions)


def model_fn_builder(albert_config,init_checkpoint,learning_rate,
                     num_train_steps,num_warmup_steps,aspects_char,optimizer='adamw'):
    def model_fn(features,labels,mode,params):
        tf.logging.info('*** Features ***')
        for name in sorted(features.keys()):
            tf.logging.info(' name = %s, shape = %s' % (name,features[name].shape))

        input_ids = features['input_ids']
        input_mask = features['input_mask']
        segment_ids = features['segment_ids']
        label_ids = features['label_ids']

        is_training = (mode == tf_estimator.estimator.ModeKeys.TRAIN)
        (total_loss, probabilities, predictions) = \
            create_model(albert_config, is_training, input_ids, input_mask,
                         segment_ids, label_ids,aspechts_char=aspects_char)

        tvars = tf.trainable_variables()
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                       init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        output_spec = None
        if mode == tf_estimator.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer(total_loss,
                                                     learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu=False)
            output_spec = tf_estimator.estimator.EstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
            )
        elif mode == tf_estimator.estimator.ModeKeys.EVAL:
            def metric_fn(loss,label_ids,logits):
                accuracy = tf.metrics.accuracy(
                    labels=label_ids,
                    predictions=predictions,
                )
                return {
                    'eval_accuracy':accuracy,
                    'eval_loss':loss,
                }

            eval_metrics = metric_fn(total_loss,label_ids,predictions)
            output_spec = tf_estimator.estimator.EstimatorSpec(
                mode = mode,
                loss = total_loss,
                eval_metric_ops=eval_metrics
            )
        else:
            output_spec = tf_estimator.estimator.EstimatorSpec(
                mode = mode,
                predictions = {
                    'probabilities':probabilities,
                    'predictions':predictions
                }
            )

        return output_spec
    return model_fn




