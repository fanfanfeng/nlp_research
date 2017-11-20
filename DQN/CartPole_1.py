import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque


# 超参数
# reward 衰减值
GAMMA = 0.9
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
REPLAY_SIZE = 10000
BATCH_SIZE = 32

class DQN():
    def __init__(self,env):
        self.replay_buffer = deque()

        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n

        self.create_Q_network()
        self.create_training_method()

        #初始化Session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    #创建Q网络
    def create_Q_network(self):
        with tf.variable_scope("input_layer"):
            self.state_input = tf.placeholder(tf.float32,[None,self.state_dim])

        with tf.variable_scope("hidden_layer"):
            W1 = tf.get_variable('W1',shape=[self.state_dim,20],initializer=tf.truncated_normal_initializer)
            b1 = tf.get_variable("b1",shape=[20],initializer=tf.truncated_normal_initializer)
            h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1)

        with tf.variable_scope('Q_value_layer'):
            W2 = tf.get_variable('W2',shape=[20,self.action_dim],initializer=tf.truncated_normal_initializer)
            b2 = tf.get_variable('b2',shape=[self.action_dim],initializer=tf.truncated_normal_initializer)
            self.Q_value = tf.matmul(h_layer,W2) + b2


    #创建训练方法
    def create_training_method(self):
        self.action_input = tf.placeholder(tf.float32,[None,self.action_dim])
        self.y_input = tf.placeholder(tf.float32,[None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices=1)
        #平方差作为loss
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)



    #存储信息
    def perceive(self,state,action,reward,next_state,done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state,one_hot_action,reward,next_state,done))

        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()


    #训练网络
    def train_Q_network(self):
        self.time_step += 1

        minibatch = random.sample(self.replay_buffer,BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
        for i in range(0,BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input:y_batch,
            self.action_input:action_batch,
            self.state_input:state_batch
        })

    #输出带随机的动作
    def egreedy_action(self,state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input:[state]
        })[0]

        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/10000
            return random.randint(0,self.action_dim - 1)
        else:
            return np.argmax(Q_value)
    #输出动作
    def action(self,state):
        return np.argmax(self.Q_value.eval(feed_dict = {
            self.state_input:[state]
        })[0])

ENV_NAME = "CartPole-v0"
EPISODE = 10000
STEP = 300
TEST = 10

def main():
    env = gym.make(ENV_NAME)
    agent = DQN(env)

    for episode in range(EPISODE):
        state = env.reset()

        for step in range(STEP):
            env.render()
            action = agent.egreedy_action(state)
            next_state,reward,done,_ = env.step(action)

            reward_target = reward #-1 if done else 0.1
            agent.perceive(state,action,reward_target,next_state,done)
            state = next_state
            if done:
                break

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

            avg_reward = total_reward/TEST
            print("Episode:",episode,'Evaluation Aveaget Reward:',avg_reward)
            if avg_reward >= 200 :
                break

if __name__ == '__main__':
    main()
