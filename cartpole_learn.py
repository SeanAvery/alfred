import gym
import math
import numpy as np
from time import sleep

class CartPoleLearner():
    def __init__(self, opts):
        print(opts)
        self.num_episodes = opts['num_episodes']
        self.num_win_ticks = opts['num_win_ticks']
        self.env = gym.make('CartPole-v0')
        self.angle_max = 0

    def reduce(self, observation):
        print(observation[2])
        degrees = math.degrees(observation[2])
        if degrees > self.angle_max:
            self.angle_max = degrees
        print('degrees', degrees)
        print(self.angle_max)
        return True

    def run(self):
        count = 0
        self.env.reset()

        for i in range(self.num_episodes):
            count +=1
            # action = self.env.action_space.sample()
            # print('action', action)
            observation, reward, done, info = self.env.step(0)
            thing = self.reduce(observation)
            self.env.render()
            sleep(0.1)
            if done:
                print('failed after {0} samples'.format(count))
                break

if __name__ == '__main__':
    options = {
        'num_episodes': 100,
        'num_win_ticks': 200,
        'num_theta_buckets': 5,
        'num_theta_prime_buckets': 5
    }

    learner = CartPoleLearner(options)
    learner.run()
