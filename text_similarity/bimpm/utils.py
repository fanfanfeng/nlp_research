# create by fanfan on 2019/12/17 0017
import tensorflow as tf

def Norm(v,axis,keepdims):
    n = tf.norm(v,axis=axis,keepdims=keepdims)
    return tf.maximum(n, 1e-6)
def cosine(v1,v2):
    m = tf.matmul(v1,tf.transpose(v2,[0,2,1]))
    n = Norm(v1,axis=2,keepdims=True) * Norm(v2,axis=2,keepdims=True)
    cosine = m /n
    return cosine

def full_matching(metric,vec,w,num_perspective):
    w = tf.expand_dims(tf.expand_dims(w,0),2)
    metric = w * tf.stack([metric] *num_perspective,axis=1 )
    vec = w * tf.stack([vec] * num_perspective,axis=1)

    m = tf.matmul(metric,tf.transpose(vec,[0,1,3,2]))
    n = Norm(metric,axis=3,keepdims=True) * Norm(metric,axis=3,keepdims=True)
    cosine = tf.transpose(m/n,[0,2,3,1])
    return cosine

def maxpool_full_matching(v1,v2,w,num_perspective):
    cosine = full_matching(v1,v2,w,num_perspective)
    max_value = tf.reduce_max(cosine,axis=2,keepdims=True)
    return max_value

def dropout_layer(input_reps,dropout_keep_rate,is_training=True):
    if is_training:
        output_repr = tf.nn.dropout(input_reps,dropout_keep_rate)
    else:
        output_repr = input_reps
    return output_repr


