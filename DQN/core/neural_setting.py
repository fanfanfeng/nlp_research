# create by fanfan on 2017/10/31 0031
class CartPole_setting():
    # 参数设置
    logdir = 'model/'
    test_per_epoch = 100
    learning_rate = 0.0001
    gamma = 0.9
    replay_memeory_size = 10000
    batch_size = 32
    initial_epsilon = 0.5
    final_epsilon = 0.01
    decay_factor = 0.99
    explore_policy = None
    save_per_step = 1000
    double_q = False

class FlappyBird_setting():
    # 参数设置
    logdir = 'model_bird/'
    test_per_epoch = 100
    learning_rate = 1e-6
    gamma = 0.9
    replay_memeory_size = 10000
    batch_size = 100
    initial_epsilon = 1
    final_epsilon = 0.0
    decay_factor = 0.9999
    explore_policy = None
    save_per_step = 1000
    double_q = False