# create by fanfan on 2019/6/26 0026

# https://www.codercto.com/a/31281.html
# https://www.cnblogs.com/mbcbyq-2137/p/10044837.html

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets(r'E:\tensorflow_data\mnist',one_hot=True)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,784],name='myInput')
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)
tf.identity(y, name="myOutput")
y_ = tf.placeholder(tf.float32,[None,10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),1))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()


for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    train_step.run({x:batch_xs,y_:batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels}))

# simple save
tf.saved_model.simple_save(sess,'output/simple/model',inputs={"myInput":x},outputs={'myOutput':y})

#compex save
builder = tf.saved_model.builder.SavedModelBuilder("output/tag/model1")
signature = tf.saved_model.predict_signature_def(inputs={'myInput':x},outputs={'myOutput':y})
builder.add_meta_graph_and_variables(sess=sess,
                                     tags=[tf.saved_model.tag_constants.SERVING],
                                     signature_def_map={'signature':signature})
builder.save()



# loader

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess,[tf.saved_model.tag_constants.SERVING],"output/tag/model1")
    graph = tf.get_default_graph()

    input = np.expand_dims(mnist.test.images[0],0)
    x = sess.graph.get_tensor_by_name('myInput:0')
    y = sess.graph.get_tensor_by_name("myOutput:0")
    batch_xs,batch_ys = mnist.test.next_batch(1)
    scores = sess.run(y,feed_dict={x:batch_xs})

    print('predict: %d,actual:%d' % (np.argmax(scores,1),np.argmax(batch_ys,1)))

