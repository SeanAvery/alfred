import gym
from gym import wrappers

class GymRunner:
    def __init__(self, timsteps=10000, monitor_dir='results/', num_episodes=1000):
        self.timesteps = timesteps
        self.num_episodes = num_episodes
        self.env = gym.make('CartPole-v0)')
        self.env = wrappers.Monitor(self.env, monitor_dir, force=True)

    def run(self):
        state = self.env.reset()

        total_reward = 0

        for i in range(num_episodes):
            print('episode', i)
