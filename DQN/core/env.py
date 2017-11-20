# create by fanfan on 2017/10/31 0031
# Env-related abstractions
import gym
from DQN.core.game import wrapped_flappy_bird
import numpy as np
import cv2
class Env():
    def step(self,action_index):
        '''
        :param action_index: 
        :return:  A tuple:(state,reward,terminal,info)
        '''
        raise NotImplementedError

    def reset(self):
        '''
        :return: state 
        '''
        raise NotImplementedError

    def render(self):
        '''
        render env,like show ganme screen
        :return: 
        '''
        raise NotImplementedError

class CartPoleEnv(Env):
    def __init__(self,monitor=False):
        self.env = gym.make('CartPole-v0')

    def step(self,action_index):
        s,r,t,i = self.env.step(action_index)
        return s,r,t,i

    def reset(self):
        return self.env.reset()

    def render(self):
        self.env.render()


# https://lufficc.com/blog/reinforcement-learning-and-implementation
image_size = 80
class FlayBirdEnv(Env):
    def __init__(self):
        self.env = wrapped_flappy_bird.GameState()
        self.pre_image_data = None

    def step(self,action_index):
        # image shape:(288,512,3)
        image_data,reward,terminal = self.env.frame_step(self.get_action(action_index))
        image_data = self.process_image_data(image_data)
        if self.pre_image_data is None:
            state = np.stack((image_data,image_data,image_data,image_data),axis=2)
        else:
            image_data = np.reshape(image_data,(image_size,image_size,1))
            state = np.append(self.pre_image_data[:,:,1:],image_data,axis=2)
        self.pre_image_data = state
        return  state,reward,terminal,{}

    def reset(self):
        self.env.reset()
        image_data,reward,terminal = self.env.frame_step(self.get_action(0))
        image_data = self.process_image_data(image_data)
        state = np.stack((image_data,image_data,image_data,image_data),axis=2)
        self.pre_image_data = state
        return state

    def render(self):
        pass

    def process_image_data(self,image_data):
        #image_data = image_data[:,:410,:]
        image_data = cv2.cvtColor(cv2.resize(image_data,(image_size,image_size)),cv2.COLOR_BGR2GRAY)

        _,image_data = cv2.threshold(image_data,1,255,cv2.THRESH_BINARY)
        return image_data


    def get_action(self,action_index):
        action = np.zeros(2)
        action[action_index] = 1
        return action
