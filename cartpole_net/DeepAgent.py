from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

import numpy as np
import random

import gym
from gym import wrappers

class DeepAgent():
    def __init__(self, opts):
        self.opts = opts
        self.monitor_dir = opts['monitor_dir']
        self.num_episodes = opts['num_episodes']
        self.state_size = 4
        self.action_size = 2

        self.memory = []

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
        model.add(Dense(12, activation='relu', input_dim=1))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(2))
        model.compile(Adam(lr=0.001), 'mse')

        self.model = model

    def replay(self):
        if len(self.memory) < self.batch_size:
            print('not enough samples')
            return 0

        else:
            mini_batch = random.sample(self.memory, self.batch_size)
            print('mini_batch', mini_batch)
            for old_state, action, reward, new_state, done in mini_batch:
                if not done:
                    target = (reward + self.gamma * np.amax(self.model.predict(new_state)) )

                target_f = self.model.predict(old_state)
                target_f[action] = target
                self.model.fit(old_state, target_f, epochs=1, verbose=0)

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def train(self):
        self.run(do_train=True)

    '''
        SIMULATION
    '''

    def build_gym(self):
        self.env = gym.make('CartPole-v0')

    def choose_action(self, state):
        if np.random.rand() <= self.epsilon:
            print('random action')
            return random.randrange(self.action_size)
        else:
            print('neural network predict')
            action = np.argmax(self.model.predict(state))
            print('action', action)
            return action

    def run(self, do_train=False):
        for episode in range(self.num_episodes-98):
            old_state = np.array(self.env.reset())
            done=False
            total_reward = 0

            while not done:
                action = self.choose_action(old_state)
                print('action', action)
                new_state, reward, done, info = self.env.step(action)
                new_state = np.array(new_state)
                total_reward += reward

                if do_train:
                    self.memory.append((old_state, action, reward, new_state, done))

                new_state = old_state

                if done:
                    break

            if do_train:
                self.replay()


if __name__ == '__main__':
    options = {
        'monitor_dir': 'results/',
        'num_episodes': 1000
    }
    agent = DeepAgent(options)
    agent.build_model()
    agent.build_gym()
    agent.train()
