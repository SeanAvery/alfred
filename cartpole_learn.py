import gym
import math
import numpy as np
from time import sleep

class CartPoleLearner():
    def __init__(self, opts):

        # cartpole simulation config
        self.num_episodes = opts['num_episodes']
        self.num_win_ticks = opts['num_win_ticks']
        self.env = gym.make('CartPole-v0')

        # state reducer
        self.buckets = opts['buckets']
        self.upper_bounds = [self.env.observation_space.high[0], 3.4, self.env.observation_space.high[2], 3.4]
        self.lower_bounds = [self.env.observation_space.low[0], -3.4, self.env.observation_space.low[2], -3.4]

        # q-learn
        self.q_table = np.zeros(self.buckets + (self.env.action_space.n,))
        self.min_epsilon = opts['min_epsilon']
        self.min_alpha = opts['min_alpha']
        self.gamma = opts['gamma'] # we do not car how long it takes

    def reduce(self, obs):
        ratios = [(obs[i] + abs(self.lower_bounds[i])) / (abs(self.upper_bounds[i]) - self.lower_bounds[i]) for i in range(len(obs))]
        reduced_obs = [round((self.buckets[i] - 1) * ratios[i]) for i in range(len(obs))]
        return reduced_obs

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample()

    def calc_epsilon(self, time):
        # linear decay w.r.t. time
        epsilon = 1 - time/100
        return max(self.min_epsilon, epsilon)

    def calc_alpha(self, time):
        # linear decay w.r.t. time
        alpha = 1 = time/100
        return max(self.min_alpha, epsilon)
    def run(self):
        count = 0

        for i in range(self.num_episodes):
            epsilon = self.get_epsilon(i)
            alpha = self.get_alpha(i)
            count +=1
            current_state = self.reduce(self.env.reset())
            done = False

            while not done:
                sleep(0.1)
                self.env.render()
                action = self.choose_action(current_state, epsilon)
                observation, reward, done, info = self.env.step(0)
                current_state = self.reduce(observation)
                if done:
                    print('failed after {0} samples'.format(count))
                    break

if __name__ == '__main__':
    options = {
        'num_episodes': 100,
        'num_win_ticks': 200,
        'num_theta_buckets': 5,
        'num_theta_prime_buckets': 5,
        'buckets': (1, 1, 12, 12),
        'min_epsilon': 0.1,
        'min_alpha': 0.1,
        'gamma': 1
    }

    learner = CartPoleLearner(options)
    learner.run()
