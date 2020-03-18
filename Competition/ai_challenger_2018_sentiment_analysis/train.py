# create by fanfan on 2020/3/13 0013
import sys
sys.path.append(r"/data/python_project/nlp_research")
from Competition.ai_challenger_2018_sentiment_analysis import settings
from utils.bert import tokenization
import third_models.albert_zh.modeling_google as modeling
import tensorflow as tf
import tensorflow_estimator as tf_estimator
from Competition.ai_challenger_2018_sentiment_analysis.model_builder import model_fn_builder
from Competition.ai_challenger_2018_sentiment_analysis.data_process import AiSentimentProcess,make_aspect_array,file_based_input_fn_builder
from Competition.ai_challenger_2018_sentiment_analysis.server_placeholder import serving_input_receiver_fn
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
def train():

    params = settings.ParamsModel()
    data_process = AiSentimentProcess()
    labels = data_process.get_labels()
    aspect_char = make_aspect_array()
    tokenization.validate_case_matches_checkpoint(params.do_lower_case,
                                                    settings.bert_model_init_path)

    if not params.do_train and not params.do_eval and not params.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    albert_config = modeling.AlbertConfig.from_json_file(settings.bert_model_config_path)

    if params.max_seq_length > albert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the ALBERT model "
            "was only trained up to sequence length %d" %
            (params.max_seq_length, albert_config.max_position_embeddings))

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
    run_config = tf_estimator.estimator.RunConfig(
        model_dir=settings.Model_save_path,
        session_config=session_conf
    )

    model_fn = model_fn_builder(
        albert_config=albert_config,
        init_checkpoint=settings.bert_model_init_path,
        learning_rate=params.learning_rate,
        num_train_steps=params.num_train_steps,
        num_warmup_steps=params.warmup_step,
        optimizer=params.optimizer,
        aspects_char=aspect_char
    )

    estimator = tf_estimator.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    tf.logging.info("***** Running training *****")
    tf.logging.info(" Batch size = %d", params.train_batch_size)
    tf.logging.info(" Num_steps = %d", params.num_train_steps)

    early_stopping_hook = tf_estimator.estimator.experimental.stop_if_no_increase_hook(
        estimator=estimator,
        metric_name='eval_loss',
        max_steps_without_increase= params.max_steps_without_decrease,
        eval_dir=None,
        min_steps=0,
        run_every_secs=None,
        run_every_steps=params.save_checkpoints_steps
    )

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"loss": "loss"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)


    train_input_fn = file_based_input_fn_builder(
        input_file=settings.train_tfrecord_path,
        seq_length=params.max_seq_length,
        is_training=True,
        batch_size=params.train_batch_size,
        buffer_size = params.buffer_size,
        n_label=params.n_class,
        n_sub_class=params.n_sub_class
    )
    train_spec = tf_estimator.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=params.num_train_steps,
        hooks = [early_stopping_hook,logging_hook]
    )



    eval_input_fn = file_based_input_fn_builder(
        input_file=settings.dev_tfrecord_path,
        seq_length=params.max_seq_length,
        is_training=False,
        batch_size=params.train_batch_size,
        buffer_size=params.buffer_size,
        n_label=params.n_class,
        n_sub_class=params.n_sub_class
    )

    exporter = tf_estimator.estimator.BestExporter(
        name="best_exporter",
        serving_input_receiver_fn=serving_input_receiver_fn,
        exports_to_keep=5)  # this will keep the 5 best checkpoints

    eval_spec = tf_estimator.estimator.EvalSpec(
        input_fn=eval_input_fn,
        exporters=exporter
    )



    tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)

    estimator._export_to_tpu = False
    estimator.export_savedmodel(settings.Output_path, serving_input_receiver_fn)


if __name__ == '__main__':
    train()

