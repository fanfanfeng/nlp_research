# create by fanfan on 2019/3/26 0026
import tensorflow as tf
import third_models.albert_zh.modeling_google as modeling
import tensorflow_estimator as tf_estimator
import third_models.albert_zh.optimization as optimization




class BaseClassifyModel(object):
    def __init__(self,params):
        self.num_tags = params.num_tags

        self.vocab_size = params.vocab_size
        self.embedding_size = params.embedding_size
        self.dropout = params.dropout


    def get_setence_length(self, data):
        used = tf.sign(tf.abs(data))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length

    def create_logits(self,input_x,dropout,already_embedded=False,real_sentence_length=None):
        with tf.variable_scope("model_define",reuse=tf.AUTO_REUSE) as scope:
            if already_embedded:
                input_embeddings = input_x
                real_sentence_length = real_sentence_length
            else:
                with tf.variable_scope('embeddings_layer'):
                    word_embeddings = tf.get_variable('word_embeddings', [self.vocab_size, self.embedding_size])
                    input_embeddings = tf.nn.embedding_lookup(word_embeddings, input_x)
                    real_sentence_length = self.get_setence_length(input_x)

            with tf.variable_scope('classify_layer'):
                output_layer = self.classify_layer(input_embeddings,dropout,real_sentence_length)


            with tf.name_scope("output"):
                logits = tf.layers.dense(output_layer, self.num_tags)
        return logits



    def classify_layer(self, input_embedding,dropout,real_sentence_length=None):
        """Implementation of specific classify layer"""
        return input_embedding

    def loss_layer(self,labels,logits):
        with tf.variable_scope('loss'):
            total_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=logits))
            mean_loss = tf.reduce_mean(total_loss)
        return total_loss,mean_loss

    def predict_layer(self,logits):
        with tf.variable_scope("prediction"):
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            probabilities = tf.nn.softmax(logits, axis=-1)
        return predictions,probabilities

    def create_model(self,input_ids,labels,is_training,albert_config=None, input_mask=None, segment_ids=None,
                     use_one_hot_embeddings=False):

        if albert_config != None:
            model = modeling.AlbertModel(
                config=albert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings)

            tf.logging.info("using pooled output")
            input_embbed = model.get_pooled_output()

            sentence_len = self.get_setence_length(input_ids)
            logits = self.create_logits(input_embbed,self.dropout,already_embedded=True,real_sentence_length=sentence_len)
        else:
            logits = self.create_logits(input_ids,self.dropout)


        total_loss,mean_loss = self.loss_layer(logits=logits,labels=labels)
        predictions, probabilities = self.predict_layer(logits)
        return (total_loss, mean_loss, probabilities, predictions)




    def create_estimator_fn(self,num_train_steps,num_warmup_steps,learning_rate,albert_config=None,init_checkpoint=None,
                     use_one_hot_embeddings=False,optimizer='adamw'):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features,labels,mode,params):
            """The `model_fn` for TPUEstimator."""
            tf.logging.info('*** Features ***')
            for name in sorted(features.keys()):
                tf.logging.info(" name = %s ,shape = %s" % (name,features[name].shape))

            is_training = (mode == tf_estimator.estimator.ModeKeys.TRAIN)
            input_ids = features['input_ids']
            label_ids = features['label_ids']
            if albert_config != None:
                input_mask = features['input_mask']
                segment_ids = features['segment_ids']
            else:
                input_mask = None
                segment_ids = None



            (total_loss,per_example_loss,probabilities,predictions) = \
                self.create_model(input_ids,labels=label_ids,is_training=is_training,
                                  albert_config=albert_config,input_mask=input_mask,
                                  segment_ids=segment_ids,use_one_hot_embeddings=use_one_hot_embeddings)

            tvars = tf.trainable_variables()

            if albert_config != None:
                initialized_variable_names = {}
                if init_checkpoint:
                    (assignment_map,initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,init_checkpoint)
                    tf.train.init_from_checkpoint(init_checkpoint,assignment_map)

                tf.logging.info("**** Trainable Variables ****")
                for var in tvars:
                    init_string = ""
                    if var.name in initialized_variable_names:
                        init_string = ", *INIT_FROM_CKPT*"
                    tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                                    init_string)

            output_spec = None
            if mode == tf_estimator.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(total_loss,
                                                         learning_rate,
                                                         num_train_steps,
                                                         num_warmup_steps,use_tpu=False)
                output_spec = tf_estimator.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                )
            elif mode == tf_estimator.estimator.ModeKeys.EVAL:
                def metric_fn(per_example_loss,label_ids,logits):
                    accuracy = tf.metrics.accuracy(
                        labels=label_ids,predictions=predictions,
                    )
                    loss = tf.metrics.mean(
                        values=per_example_loss
                    )
                    return {
                        'eval_accuracy':accuracy,
                        'eval_loss':loss,
                    }

                eval_metrics = metric_fn(per_example_loss,label_ids,predictions)
                output_spec = tf_estimator.estimator.EstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metric_ops=eval_metrics
                )
            else:
                output_spec = tf_estimator.estimator.EstimatorSpec(
                    mode=mode,
                    predictions={
                        'probabilities':probabilities,
                        'predictions':predictions,
                    },
                )

            return output_spec

        return model_fn


