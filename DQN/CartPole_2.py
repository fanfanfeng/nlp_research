__author__ = 'fanfan'
class Model():
    def __init__(self,num_outputs):
        self.num_outputs = num_outputs

    def definition(self):
        raise NotImplementedError

    def get_num_outputs(self):
        return self.num_outputs

# Env-related abstractions
class Env():
    def step(self,action_index):
        raise NotImplementedError
    # 返回一个元组：(state, reward, terminal, info)，state 是游戏当前状态，如屏幕像素；reward 是奖励，可以为负数；
    # terminal 代表游戏是否结束；info 代表一些其他信息，可有可无
    def reset(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

import numpy as np
import tensorflow as tf
import random
import numpy as np
import time
import sys
import collections

# 参数设置
learning_rate = 0.001
gamma = 0.9
replay_memory_size = 10000
batch_size = 32
initial_epsilon = 0.5
final_epsilon = 0.01
decay_factor = 1
logdir = 'model_save/'
save_per_step = 1000
test_per_epoch = 100

class DeepQNetwork(object):
    def __init__(self,model,env):
        self.model = model
        self.env = env
        self.num_actions = model.get_num_outputs()

        self._init_setting()
        self.define_q_network()

    def _init_setting(self):
        self.learning_rate = learning_rate
        self.optimizer = tf.train.AdamOptimizer
        self.gamma = gamma
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_factor = decay_factor
        self.logdir = logdir
        self.test_per_epoch = test_per_epoch
        self.save_per_step = save_per_step

        self.replay_memeory = collections.deque()
        self.replay_memeory_size = replay_memory_size
        self.batch_size = batch_size

        self.saver = tf.train.Saver()

    def define_q_network(self):
        self.input_states,self.q_values = self.model.definition()
        self.input_actions = tf.placeholder(tf.float32,[None,self.num_actions])

        # 目标 Q_values
        self.input_q_values = tf.placeholder(tf.float32,[None])

        action_q_values = tf.reduce_sum(tf.multiply(self.q_values,self.input_actions), reduction_indices=1)
        self.global_step = tf.Variable(0,trainable=False)
        self.cost = tf.reduce_mean(tf.square(self.input_q_values - action_q_values))
        self.optimizer = self.optimizer(self.learning_rate).minimize(self.cost,global_step = self.global_step)

        tf.summary.scalar('cost',self.cost)
        tf.summary.scalar('reward',tf.reduce_mean(action_q_values))


    def egreedy_action(self,state):
        if random.random() <= self.epsilon:
            action_index = random.randint(0,self.num_actions - 1)
        else:
            action_index = self.action(state)

        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.decay_factor
        return action_index

    def action(self,state):
        q_values = self.q_values.eval(feed_dict={self.input_states:[self]})[0]
        return np.argmax(q_values)

    def do_train(self,epoch):
        # 随机选择一个batch
        mini_batchse = random.sample(self.replay_memeory,self.batch_size)
        state_batch = [batch[0] for batch in mini_batchse]
        action_batch = [batch[1] for batch in mini_batchse]
        reward_batch = [batch[2] for batch in mini_batchse]
        next_state_batch = [batch[3] for batch in mini_batchse]

        # 目标Q_value
        target_q_values = self.q_values.eval(feed_dict={
            self.input_states:next_state_batch
        })
        input_q_values = []
        for i in range(len(mini_batchse)):
            terminal = mini_batchse[i][4]
            if terminal:
                input_q_values.append(reward_batch[i])
            else:
                input_q_values.append(reward_batch[i] + self.gamma * np.max(target_q_values[i]))

        feed_dict = {
            self.input_actions:action_batch,
            self.input_states:state_batch,
            self.input_q_values:input_q_values
        }
        self.optimizer.run(feed_dict = feed_dict)

        current_step = self.global_step.eval()
        if self.saver is not None and epoch > 0 and current_step % self.save_per_step == 0:
            summary = self

def main():
    model
    sess = tf.InteractiveSession()


