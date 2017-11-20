__author__ = 'fanfan'

import numpy as np
import tensorflow as tf
import random
import collections
import sys



class DeepQNetwork(object):
    def __init__(self,model,env,settings,explore_policy = None):
        self.model = model
        self.env = env
        self.num_actions = model.get_num_outputs()
        self.settings = settings
        self.explore_policy = explore_policy

        self._init_setting()
        self.define_q_network()

        # reward of every epoch
        self.rewards = []
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(tf.global_variables())

        self.summayies = tf.summary.merge_all()
        self.log_writer = tf.summary.FileWriter(self.logdir,self.sess.graph)

        self._check_model()

    def _init_setting(self):
        self.learning_rate = self.settings.learning_rate
        self.optimizer = tf.train.AdamOptimizer
        self.gamma = self.settings.gamma
        self.epsilon = self.settings.initial_epsilon
        self.final_epsilon = self.settings.final_epsilon
        self.decay_factor = self.settings.decay_factor
        self.logdir = self.settings.logdir
        self.test_per_epoch = self.settings.test_per_epoch
        self.save_per_step = self.settings.save_per_step

        self.replay_memeory = collections.deque()
        self.replay_memeory_size = self.settings.replay_memeory_size
        self.batch_size = self.settings.batch_size
        self.global_step = tf.Variable(0, trainable=False)
        #self.learning_rate = tf.train.exponential_decay(self.learning_rate,self.global_step,self.save_per_step,0.8,staircase=True)

        if self.explore_policy is None:
            self.explore_policy =  lambda  epsilon:random.randint(0,self.num_actions -1)



    def _check_model(self):
        if self.logdir is not None:
            if not self.logdir.endswith("/"):
                self.logdir += "/"
            checkpoint_state = tf.train.get_checkpoint_state(self.logdir)
            if checkpoint_state and checkpoint_state.model_checkpoint_path:
                path = checkpoint_state.model_checkpoint_path
                self.saver.restore(self.sess,path)
                print("Restore form {} successfully".format(path))
            else:
                print("No checkpoint")
                self.sess.run(tf.global_variables_initializer())


    def define_q_network(self):
        self.input_states,self.q_values = self.model.definition()
        self.input_actions = tf.placeholder(tf.float32,[None,self.num_actions],name='actions')

        # 目标 Q_values
        self.input_q_values = tf.placeholder(tf.float32,[None],name='target_q_values')

        action_q_values = tf.reduce_sum(tf.multiply(self.q_values,self.input_actions), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.input_q_values - action_q_values),name='cost')
        self.optimizer = self.optimizer(self.learning_rate).minimize(self.cost,global_step = self.global_step)

        tf.summary.scalar('cost',self.cost)
        tf.summary.scalar('reward',tf.reduce_mean(action_q_values))


    def egreedy_action(self,state):
        if random.random() <= self.epsilon:
            action_index = self.explore_policy(self.epsilon)
        else:
            action_index = self.action(state)

        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.decay_factor
        return action_index

    def action(self,state):
        q_values = self.q_values.eval(feed_dict={self.input_states:[state]})[0]
        return np.argmax(q_values)

    def q_values_function(self,states):
        return self.q_values.eval(feed_dict={self.input_states:states})


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
            summary = self.sess.run(self.summayies,feed_dict=feed_dict)
            self.log_writer.add_summary(summary,current_step)
            self.saver.save(self.sess,self.logdir+"dqn",self.global_step)


    def train(self,num_epoches):
        for epoch in range(num_epoches):
            epoch_rewards = 0
            state = self.env.reset()
            for step in range(9999999):
                self.env.render()
                action_index = self.egreedy_action(state)
                next_state,reward,terminal,info = self.env.step(action_index)

                one_hot_action = np.zeros([self.num_actions])
                one_hot_action[action_index] = 1

                self.replay_memeory.append((state,one_hot_action,reward,next_state,terminal))
                if len(self.replay_memeory) > self.replay_memeory_size:
                    self.replay_memeory.popleft()

                if len(self.replay_memeory) > self.batch_size:
                    self.do_train(epoch)

                state = next_state
                epoch_rewards += reward
                if terminal:
                    self.rewards.append(epoch_rewards)
                    break
            if epoch > 0 and epoch % 100 == 0:
                print("第{}步，当前的reward:{}，".format(epoch,epoch_rewards))

            if epoch >0 and epoch % self.test_per_epoch == 0:
                self.test(epoch,max_step_per_test=99999)


    def test(self,epoch,num_tests=10,max_step_per_test=300):
        totoal_rewards = 0
        print("Testing....")
        sys.stdout.flush()
        for _ in range(num_tests):
            state = self.env.reset()
            for step in range(max_step_per_test):
                action = self.action(state)
                state,reward,terminal,info = self.env.step(action)
                totoal_rewards += reward
                if terminal:
                    break
        average_reward = totoal_rewards/num_tests
        print("epoch {:5} average_reward:{}".format(epoch,average_reward))




