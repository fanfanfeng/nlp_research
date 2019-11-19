# create by fanfan on 2019/10/17 0017
#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
## https://www.cnblogs.com/pinard/p/9923859.html ##
## 强化学习(十二) Dueling DQN ##

import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# hyper Paramters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FIANL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch
REPLACE_TARGET_FREQ = 10 # frequency to update target Q network

class DQN():
    # DQN Agent
    def __init__(self,env):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        # init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # input layer
        self.state_input = tf.placeholder(dtype=tf.float32,shape=[None,self.state_dim])

        # network weights
        with tf.variable_scope('current_net'):
            W1 = self.weight_variable([self.state_dim,20])
            b1 = self.bias_variable([20])


            # hidden layers
            h_layers = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)

            # hidden layer for state value
            with tf.variable_scope('Value'):
                W21 = self.weight_variable([20,1])
                b21 = self.bias_variable([1])
                self.V = tf.matmul(h_layers,W21) + b21

            with tf.variable_scope('Advantage'):
                W22 = self.weight_variable([20,self.action_dim])
                b22= self.bias_variable([self.action_dim])
                self.A = tf.matmul(h_layers,W22) + b22

            self.Q_value = self.V + (self.A - tf.reduce_mean(self.A,axis=1,keep_dims=True))



        with tf.variable_scope('target_net'):
            W1_t = self.weight_variable([self.state_dim,20])
            b1_t = self.bias_variable([20])


            # hidden layers
            h_layers_t = tf.nn.relu(tf.matmul(self.state_input,W1_t) + b1_t)

            # hidden layer for state value
            with tf.variable_scope('Value'):
                W2v = self.weight_variable([20, 1])
                b2v = self.bias_variable([1])
                self.VT = tf.matmul(h_layers, W2v) + b2v

            with tf.variable_scope('Advantage'):
                W2a = self.weight_variable([20, self.action_dim])
                b2a = self.bias_variable([self.action_dim])
                self.AT = tf.matmul(h_layers, W2a) + b2a

            self.target_Q_value = self.VT + (self.AT - tf.reduce_mean(self.AT, axis=1, keep_dims=True))

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='current_net')

        with tf.variable_scope("soft_replacement"):
            self.target_replace_op = [tf.assign(t,e) for t,e in zip(t_params,e_params)]


    def create_training_method(self):
        self.action_input = tf.placeholder(dtype=tf.float32,shape=[None,self.action_dim]) # one hot presentation
        self.y_input = tf.placeholder(dtype=tf.float32,shape=[None])

        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)


    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1

        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()


        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()


    def train_Q_network(self):
        self.time_step += 1

        # step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)

        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch =[data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]


        # step 2: calculate y
        y_batch = []
        Q_value_batch = self.target_Q_value.eval(feed_dict={self.state_input:next_state_batch})

        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input: state_batch
        })

    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input:[state]
        })[0]

        if random.random() <= self.epsilon:
            self.epsilon = (INITIAL_EPSILON - FIANL_EPSILON)/10000
            return random.randint(0,self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FIANL_EPSILON) / 10000
            return np.argmax(Q_value)

    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {
            self.state_input:[state]
        })[0])


    def update_target_q_network(self,episode):
        # update target Q network
        if episode % REPLACE_TARGET_FREQ == 0:
            self.session.run(self.target_replace_op)


    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01,shape=shape)
        return tf.Variable(initial)


# ---------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000 # Episode limitation
STEP = 300
TEST = 10 # The number of experiment test every 100 episode

def main():
    # initialize OpenAi Gym env and dqn agent
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()

        # train
        for step in range(STEP):
            action = agent.egreedy_action(state)
            next_state,reward,done,_ = env.step(action)

            # Define reward for agent
            reward = - 1 if done else 0.1
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.action(state)
                    state,reward,done,_ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward /TEST
            print("episode:",episode,"Evaluation Average Reward:",ave_reward)

        agent.update_target_q_network(episode)
if __name__ == '__main__':
    main()