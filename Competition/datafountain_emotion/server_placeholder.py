# create by fanfan on 2020/3/17 0017
import tensorflow as tf
from Competition.datafountain_emotion.settings import ParamsModel
param = ParamsModel()
def serving_input_receiver_fn():
    """Serving input_fn that builds features from placeholders
    Returns
    -------
    tf.estimator.export.ServingInputReceiver
    """
    # feature = InputFeatures(
    # input_ids=input_ids,
    # input_mask=input_mask,
    # segment_ids=segment_ids,
    # label_ids=label_id,
    # seq_length = seq_length,
    # is_real_example=True)

    input_ids = tf.placeholder(dtype=tf.int32, shape=[None, param.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[None, param.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, param.max_seq_length], name='segment_ids')
    label_ids = tf.placeholder(dtype=tf.int32, name='label_ids')
    receiver_tensors = {'input_ids': input_ids,
                            'input_mask': input_mask,
                            'segment_ids': segment_ids,
                            'label_ids': label_ids
                        }
    features = {'input_ids': input_ids,
                'input_mask': input_mask,
                 'segment_ids': segment_ids,
                'label_ids': label_ids
                }
    #return features
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)