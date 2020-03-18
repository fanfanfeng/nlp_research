# create by fanfan on 2020/3/17 0017
import tensorflow as tf

def serving_input_receiver_fn(max_sentence_len,n_class,n_subclass):
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

    input_ids = tf.placeholder(dtype=tf.int32, shape=[None, max_sentence_len], name='input_ids')
    input_mask = tf.placeholder(dtype=tf.int32, shape=[None, max_sentence_len], name='input_mask')
    segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, max_sentence_len], name='segment_ids')
    label_ids = tf.placeholder(dtype=tf.int32, shape=[None,n_class,n_subclass], name='label_ids')
    receiver_tensors = {'input_ids': input_ids,
                            'input_mask': input_mask,
                            'segment_ids': segment_ids,
                            'label_ids':label_ids}
    features = {'input_ids': input_ids,
                            'input_mask': input_mask,
                            'segment_ids': segment_ids,
                            'label_ids':label_ids}
    return features
    #return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)