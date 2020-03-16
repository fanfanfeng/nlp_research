# create by fanfan on 2020/3/13 0013
import sys
sys.path.append(r"/data/python_project/nlp_research")
from Competition.datafountain_emotion import settings
from utils.bert import tokenization
import third_models.albert_zh.modeling_google as modeling
import tensorflow as tf
import tensorflow_estimator as tf_estimator
from Competition.datafountain_emotion.model_builder import model_fn_builder
from Competition.datafountain_emotion.data_process import DataFountainEmotionProcess
from utils.bert.tfrecord_utils import file_based_input_fn_builder
from utils.bert.utils import serving_input_fn,serving_input_receiver_fn
import os

def train():
    params = settings.ParamsModel()
    data_process = DataFountainEmotionProcess()
    labels = data_process.get_labels()
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
        num_labels=len(labels),
        init_checkpoint=settings.bert_model_init_path,
        learning_rate=params.learning_rate,
        num_train_steps=params.num_train_steps,
        num_warmup_steps=params.warmup_step,
        use_one_hot_embeddings=params.use_one_hot_embeddings,
        optimizer=params.optimizer
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
    #tensors_to_log = {"probabilities": "softmax_tensor"}
    #logging_hook = tf.train.LoggingTensorHook(
    #    tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = file_based_input_fn_builder(
        input_file=settings.train_tfrecord_path,
        seq_length=params.max_seq_length,
        is_training=True,
        drop_remainder=True,
        batch_size=params.train_batch_size,
        buffer_size = params.buffer_size
    )
    train_spec = tf_estimator.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=params.num_train_steps,
        hooks = [early_stopping_hook]
    )



    eval_input_fn = file_based_input_fn_builder(
        input_file=settings.dev_tfrecord_path,
        seq_length=params.max_seq_length,
        is_training=False,
        drop_remainder=False,
        batch_size=params.train_batch_size,
        buffer_size=params.buffer_size
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





def predict():
    params = settings.ParamsModel()
    data_process = DataFountainEmotionProcess()
    labels = data_process.get_labels()
    albert_config = modeling.AlbertConfig.from_json_file(settings.bert_model_config_path)

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9
    run_config = tf_estimator.estimator.RunConfig(
        model_dir=settings.Model_save_path,
        session_config=session_conf
    )

    model_fn = model_fn_builder(
        albert_config=albert_config,
        num_labels=len(labels),
        init_checkpoint=settings.bert_model_init_path,
        learning_rate=params.learning_rate,
        num_train_steps=params.num_train_steps,
        num_warmup_steps=params.warmup_step,
        use_one_hot_embeddings=params.use_one_hot_embeddings,
        optimizer=params.optimizer
    )

    estimator = tf_estimator.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    tf.logging.info("**** Running predict ****")
    predict_input_fn = file_based_input_fn_builder(
        input_file=settings.test_tfrecord_path,
        seq_length=params.max_seq_length,
        is_training=False,
        drop_remainder=False,
        batch_size=params.train_batch_size,
        buffer_size=params.buffer_size
    )

    result = estimator.evaluate(input_fn=predict_input_fn, steps=None)
    output_predict_file = os.path.join(settings.Output_path, 'test_results.csv')
    with tf.gfile.GFile(output_predict_file, 'w') as writer:
        num_written_lines = 0
        tf.logging.info("******* Predict results ********")
        for (i, prediction) in enumerate(result):
            probabilities = prediction['probabilites']

            output_line = "\t".join(
                str(class_probability)
                for class_probability in probabilities
            ) + "\n"

            writer.write(output_line)
            num_written_lines += 1


if __name__ == '__main__':
    train()

