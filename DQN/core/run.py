# create by fanfan on 2017/10/31 0031
import random
from DQN.core.env import CartPoleEnv,FlayBirdEnv
from DQN.core.model import SimpleNeuralNetwork,CNN
from DQN.core.dqn import DeepQNetwork
from DQN.core.neural_setting import CartPole_setting,FlappyBird_setting

def train_CartPole(train=True):
    model = SimpleNeuralNetwork([4, 24, 2])
    env = CartPoleEnv()
    qnetwork = DeepQNetwork(
        model=model, env=env, settings=CartPole_setting)
    if train:
        qnetwork.train(4000)
    else:
        runGame(env,qnetwork)

image_size = 80
def train_FlappyBirdEnv(train=True):
    model = CNN(img_w=image_size,img_h=image_size,num_outputs=2)
    env = FlayBirdEnv()

    def explor_policy(epsilon):
        a = random.random()
        if a < 0.95:
            action_index = 0
        else:
            action_index = 1
        return  action_index

    qnetwork = DeepQNetwork(model=model,env=env,settings=FlappyBird_setting,explore_policy=explor_policy)
    if train:
        qnetwork.train(100000)
    else:
        runGame(env,qnetwork)

def runGame(env, network):
    state = env.reset()
    reward_total = 0

    while True:
        env.render()
        action = network.action(state)
        state, reward, terminal, _ = env.step(action)
        reward_total += reward
        if terminal:
            state = env.reset()
            print("reward奖励值（200就说明达到目标，游戏结束，没有的话，就说明游戏失败）：",reward_total)
            reward_total = 0

if __name__ == '__main__':
    train_FlappyBirdEnv(train=False)