# create by fanfan on 2019/10/17 0017
#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################
## https://www.cnblogs.com/pinard/p/10137696.html ##
## 强化学习(十三) 策略梯度(Policy Gradient) ##
import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters
GAMMA = 0.95
LEARNING_RATE = 0.01

class Policy_Gradient():
    def __init__(self,env):
        # init some parameters
        self.time_step = 0
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.ep_obs,self.ep_as,self.ep_rs = [],[],[]
        self.create_softmax_network()

        # init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_softmax_network(self):
        # network weights
        W1 = self.weight_variable([self.state_dim,20])
        b1 = self.bias_variable([20])
        W2 = self.weight_variable([20,self.action_dim])
        b2 = self.bias_variable([self.action_dim])

        # input layer
        self.state_input = tf.placeholder(dtype=tf.float32,shape=[None,self.state_dim])
        self.tf_acts = tf.placeholder(tf.int32,[None,],name='actions_num')
        self.tf_vt = tf.placeholder(tf.float32,[None,],name='actions_value')

        # hidden layer
        h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)

        # softmax layer
        self.softmax_input = tf.matmul(h_layer,W2) + b2

        # softmax output
        self.all_act_prob = tf.nn.softmax(self.softmax_input,name='act_prob')
        self.neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.softmax_input,
            labels=self.tf_acts
        )
        self.loss = tf.reduce_mean(self.neg_log_prob * self.tf_vt)

        self.train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)


    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def choose_action(self,observation):
        prob_weights = self.session.run(self.all_act_prob,feed_dict={
            self.state_input:observation[np.newaxis,:]
        })
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())
        return action


    def store_transition(self,s,a,r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)


    def learn(self):
        discournted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0,len(self.ep_rs))):
            running_add = running_add *GAMMA +self.ep_rs[t]
            discournted_ep_rs[t] = running_add
        discournted_ep_rs -= np.mean(discournted_ep_rs)
        discournted_ep_rs /= np.std(discournted_ep_rs)

        # train on episode
        self.session.run(self.train_op,feed_dict={
            self.state_input: np.vstack(self.ep_obs),
            self.tf_acts:np.array(self.ep_as),
            self.tf_vt:discournted_ep_rs
        })

        self.ep_obs,self.ep_as,self.ep_rs = [],[],[]

# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000
STEP = 3000
TEST = 10

def main():
    # initialize task
    env = gym.make(ENV_NAME)
    agent = Policy_Gradient(env)

    for episode in range(EPISODE):
        # initialize task
        state = env.reset()

        # Train
        for step in range(STEP):
            action = agent.choose_action(state)
            next_state,reward,done,_ = env.step(action)
            agent.store_transition(state,action,reward)
            state = next_state

            if done:
                agent.learn()
                break

        # Test every 100 episodes
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    env.render()
                    action = agent.choose_action(state)
                    state,reward,done,_ = env.step(action)
                    total_reward += reward

                    if done:
                        break

            ave_reward = total_reward / TEST
            print("episode:",episode,'Evaluation Average Reward:',ave_reward)

if __name__ == '__main__':
    main()