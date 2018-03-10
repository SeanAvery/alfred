from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import gym
from gym import wrappers

class DeepAgent():
    def __init__(self, opts):
        self.opts = opts
        self.monitor_dir = opts['monitor_dir']
        self.num_episodes = opts['num_episodes']

        # hyperparameters
        self.gamma = 0.99 # discount rate
        self.epsilon = 1.0 # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.batch_size = 32


    '''
        NEURAL NETWORK
    '''
    def build_model(self):
        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=4))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(2))
        model.compile(Adam(lr=0.001), 'mse')

    '''
        SIMULATION
    '''
    def build_gym(self):
        self.env = gym.make('CartPole-v0')

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon

    def run(self):
        for episode in range(self.num_episodes-98):
            old_state = self.env.reset()
            done=False

            while not done:
                action = self.choose_action(old_state)


if __name__ == '__main__':
    options = {
        'monitor_dir': 'results/',
        'num_episodes': 1000
    }
    agent = DeepAgent(options)
    agent.build_model()
    agent.build_gym()
    agent.run()
