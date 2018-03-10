import keras.backend as K
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from tensorflow.python import debug as tf_debug

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
        # sess = K.get_session()
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # K.set_session(sess)

        model = Sequential()
        model.add(Dense(12, activation='relu', input_dim=4))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(2))
        model.compile(Adam(lr=0.001), 'mse')

        self.model = model

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        else:
            mini_batch = random.sample(self.memory, self.batch_size)
            print('mini_batch', mini_batch)
            for old_state, action, reward, new_state, done in mini_batch:
                if not done:
                    target = (reward + self.gamma * np.amax(self.model.predict(new_state)) )
                else:
                    target = reward
                target_f = self.model.predict(old_state)
                target_f[0][action] = target
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
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])

    def run(self, do_train=False, num_episodes):
        for episode in range(num_episodes):
            old_state = self.env.reset().reshape(1, self.env.observation_space.shape[0])
            done=False
            total_reward = 0

            while not done:
                action = self.choose_action(old_state)
                new_state, reward, done, info = self.env.step(action)
                new_state = new_state.reshape(1, self.env.observation_space.shape[0])
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
