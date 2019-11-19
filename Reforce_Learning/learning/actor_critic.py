# create by fanfan on 2019/10/17 0017
#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
## https://www.cnblogs.com/pinard/p/10272023.html ##
## 强化学习(十四) Actor-Critic ##

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters
GAMMA = 0.95
LEARNING_RATE = 0.01

class Actor():
    def __init__(self,env,sess):
        # init some parameters
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.create_softmax_network()

        # Init session
        self.session = sess
        self.session.run(tf.global_variables_initializer())


    def create_softmax_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim,20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20,self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        # input layer
        self.state_input = tf.placeholder(dtype=tf.float32,shape=[None,self.state_dim])
        self.tf_acts = tf.placeholder(dtype=tf.int32,shape=[None,2],name='actions_num')
        self.td_error = tf.placeholder(dtype=tf.float32,shape=None,name='td_error')

        # hidden layers
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)

        #softmax layer
        self.softmax_input = tf.matmul(h_layer,W2) + b2

        # softmax output
        self.all_act_prob = tf.nn.softmax(self.softmax_input,name='act_prob')
        self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits(logits=self.softmax_input,labels=self.tf_acts)

        self.exp = tf.reduce_mean(self.neg_log_prob * self.td_error)
        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(-self.exp)




    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


    def choose_action(self,observation):
        prob_weights = self.session.run(self.all_act_prob,feed_dict={self.state_input:observation[np.newaxis,:]})
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action

    def learn(self,state,action,td_error):
        s = state[np.newaxis,:]
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        a = one_hot_action[np.newaxis,:]

        # train on episode
        self.session.run(self.train_op,feed_dict={
            self.state_input:s,
            self.tf_acts:a,
            self.td_error:td_error
        })


EPSILON = 0.01
REPLAY_SIZE = 10000
BATCH_SIZE = 32
REPLACE_TARGET_FREQ = 10

class Critic():
    def __init__(self,env,sess):
        self.time_step = 0
        self.epsilon = EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = sess
        self.session.run(tf.global_variables_initializer())


    def create_Q_network(self):
        # network weights
        W1q = self.weight_variable([self.state_dim,20])
        b1q = self.bias_variable([20])
        W2q = self.weight_variable([20,1])
        b2q = self.bias_variable([1])

        self.state_input = tf.placeholder(dtype=tf.float32,shape=[1,self.state_dim],name='state')

        # hidden layer
        h_layerq = tf.nn.relu(tf.matmul(self.state_input,W1q) + b1q)

        # Q value layer
        self.Q_value = tf.matmul(h_layerq,W2q) + b2q


    def create_training_method(self):
        self.next_value = tf.placeholder(dtype=tf.float32,shape=[1,1],name='v_next')
        self.reward = tf.placeholder(dtype=tf.float32,shape=(),name='reward')

        with tf.variable_scope("squared_TD_error"):
            self.td_error = self.reward + GAMMA * self.next_value - self.Q_value
            self.loss = tf.square(self.td_error)

        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.epsilon).minimize(self.loss)


    def train_Q_network(self,state,reward,next_state):
        s,s_ = state[np.newaxis,:],next_state[np.newaxis,:]
        v_ = self.session.run(self.Q_value,{self.state_input:s_})
        td_error,_ = self.session.run([self.td_error,self.train_op],{
            self.state_input:s,
            self.next_value:v_,
            self.reward:reward
        })
        return td_error

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000
STEP = 3000
TEST = 10

def main():
    # initialize Open AI Gym env and dqn agent
    sess = tf.InteractiveSession()
    env = gym.make(ENV_NAME)
    actor = Actor(env,sess)
    critic = Critic(env,sess)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()

        # Train
        for step in range(STEP):
            action = actor.choose_action(state)
            next_state,reward,done,_ = env.step(action)
            td_error = critic.train_Q_network(state,reward,next_state)
            actor.learn(state,action,td_error)
            state = next_state
            if done:
                break


        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = actor.choose_action(state)
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break

            ave_reward  = total_reward/TEST
            print("episode:",episode,'Evaluation Average Reward:',ave_reward)

if __name__ == '__main__':
    main()












