# create by fanfan on 2018/8/17 0017
import tensorflow as tf


class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, scope="batch_norm"):
    with tf.variable_scope(scope):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.scope = scope

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum,
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.scope)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return tf.concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)