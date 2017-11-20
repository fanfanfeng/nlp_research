# create by fanfan on 2017/10/31 0031
import tensorflow as tf
class Model():
    def __init__(self,num_outputs,name_scope='model'):
        self.num_outputs = num_outputs
        self.name_scope = name_scope

    def definition(self):
        raise NotImplementedError

    def get_num_outputs(self):
        return self.num_outputs


class SimpleNeuralNetwork(Model):
    def __init__(self,layers,name_scope="simple_neural_network"):
        Model.__init__(self,layers[-1],name_scope)
        self.num_inputs = layers[0]
        self.layers = layers


    def definition(self):
        with tf.name_scope(self.name_scope):
            inputs = tf.placeholder(tf.float32,[None,self.num_inputs],name='inputs')
            layer = inputs
            current_layer = 2
            for i,j in zip(self.layers[:-1],self.layers[1:]):
                w = tf.Variable(tf.truncated_normal([i,j]),name='layer_'+ str(current_layer)+"_weights")
                b = tf.Variable(tf.constant(0.01,shape=[j]),name='layer_'+str(current_layer)+"_biases")

                dense = tf.matmul(layer,w) + b
                if len(self.layers) != current_layer:
                    dense = tf.nn.relu(dense)
                layer = dense
                current_layer += 1
            outputs = layer

        return inputs,outputs

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

class CNN(Model):
    def __init__(self,img_w,img_h,num_outputs,name_scope="cnn"):
        Model.__init__(self,num_outputs,name_scope)
        self.img_w = img_w
        self.img_h = img_h

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape,stddev=0.01)
        return tf.Variable(initial)

    def bias_varibale(self,shape):
        initial = tf.constant(0.01,shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x,W,stride):
        return tf.nn.conv2d(x,W,strides=[1,stride,stride,1],padding='SAME')

    def max_pool_2x2(self,x):
        return  tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    def definition(self):
        with tf.name_scope(self.name_scope):
            W_conv1 = self.weight_variable([8,8,4,32])
            b_conv1 = self.bias_varibale([32])

            W_conv2 = self.weight_variable([4,4,32,64])
            b_conv2 = self.bias_varibale([64])

            W_conv3 = self.weight_variable([3,3,64,64])
            b_conv3 = self.bias_varibale([64])

            # input_layer
            s = tf.placeholder(tf.float32,[None,self.img_w,self.img_h,4])

            with tf.variable_scope("hidden_layer_1"):
                h_conv1 = tf.nn.conv2d(s,W_conv1,strides=[1,4,4,1],padding='SAME') + b_conv1
                h_conv1 = tf.nn.relu(h_conv1)#shape=[16,16,32]

                h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
                # shape = [8,8,32]
            with tf.variable_scope("hidden_layer_2"):
                h_conv2 = tf.nn.conv2d(h_pool1,W_conv2,strides=[1,2,2,1],padding='SAME') + b_conv2
                h_conv2 = tf.nn.relu(h_conv2)
                # shape = [4.4,64]

            with tf.variable_scope("hidden_layer_3"):
                h_conv3 = tf.nn.conv2d(h_conv2,W_conv3,strides=[1,1,1,1],padding='SAME') + b_conv3
                h_conv3 = tf.nn.relu(h_conv3)
                # shape = [4,4,64]


            count = int(h_conv3.shape[1] * h_conv3.shape[2]* h_conv3.shape[3])
            with tf.variable_scope("full_connect"):
                W_fc1 = self.weight_variable([count,512])
                b_fc1 = self.bias_varibale([512])

                W_fc2 = self.weight_variable([512,2])
                b_fc2 = self.bias_varibale([2])

                h_conv3_flat = tf.reshape(h_conv3,[-1,count])

                h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)

                out = tf.matmul(h_fc1,W_fc2) + b_fc2

        return s,out



