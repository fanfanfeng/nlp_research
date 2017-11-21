__author__ = 'fanfan'
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers

sess = tf.InteractiveSession()

#tf.contrib.layers.embed_sequence
#链接：https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence
#说明：对序列数据执行embedding操作，输入[batch_size, sequence_length]的tensor，返回[batch_size, sequence_length, embed_dim]的tensor。
print("tf.contrib.layers.embed_sequence test begin:")
features = np.arange(6).reshape(2,3)
outputs = layers.embed_sequence(features,vocab_size=6,embed_dim=4)
sess.run(tf.global_variables_initializer())
print(sess.run(outputs))

#tf.strided_slice
#链接：https://www.tensorflow.org/api_docs/python/tf/strided_slice
#说明：对传入的tensor执行切片操作，返回切片后的tensor。主要参数input_, start, end, strides，strides代表切片步长
input = [
    [[1, 1, 1], [2, 2, 2]],
    [[3, 3, 3], [4, 4, 4]],
    [[5, 5, 5], [6, 6, 6]]
         ]
output1 = sess.run(tf.strided_slice(input,[1,0,0],[2,1,3],[1,1,1]))
print(output1) #[[[3 3 3]]]
output2 = sess.run(tf.strided_slice(input,[1,0,0],[2,2,3],[1,1,1]))
print(output2) #[[[3,3,3],[4,4,4]]]


#tf.contrib.seq2seq.TrainingHelper
# 链接：https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/TrainingHelper
#说明：Decoder端用来训练的函数。这个函数不会把t-1阶段的输出作为t阶段的输入，而是把target中的真实值直接输入给RNN。主要参数是inputs和sequence_length。返回helper对象，可以作为BasicDecoder函数的参数。
#例子：
#   training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
#                                                  sequence_length=target_sequence_length,
#                                                  time_major=False)

#tf.contrib.seq2seq.BasicDecoder
#链接：https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder
#说明：生成基本解码器对象
#例子：
#  cell为RNN层，training_helper是由TrainingHelper生成的对象，
#  encoder_state是RNN的初始状态tensor，
#  output_layer代表输出层，它是一个tf.layers.Layer的对象。
#  training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
#                                                 training_helper,
#                                                 encoder_state,
#                                                 output_layer)

#tf.contrib.seq2seq.dynamic_decode
#链接：https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode
#说明：对decoder执行dynamic decoding。通过maximum_iterations参数定义最大序列长度。

#tf.contrib.seq2seq.GreedyEmbeddingHelper
#链接：https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/GreedyEmbeddingHelper
#说明：它和TrainingHelper的区别在于它会把t-1下的输出进行embedding后再输入给RNN。

#tf.sequence_mask
#链接：https://www.tensorflow.org/api_docs/python/tf/sequence_mask
#说明：对tensor进行mask，返回True和False组成的tensor
mask_out =  tf.sequence_mask([1,3,2],5)
print(sess.run(mask_out))
#[[True, False, False, False, False],
#  [True, True, True, False, False],
#  [True, True, False, False, False]]


#tf.contrib.seq2seq.sequence_loss
#链接：https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/sequence_loss
#说明：对序列logits计算加权交叉熵。
#例子：
# training_logits是输出层的结果，
# targets是目标值，
# masks是我们使用tf.sequence_mask计算的结果，在这里作为权重，也就是说我们在计算交叉熵时不会把<PAD>计算进去。
#  cost = tf.contrib.seq2seq.sequence_loss(
#                                           training_logits,
#                                           targets,
#                                           masks)
sess.close()