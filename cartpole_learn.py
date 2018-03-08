import gym
import math
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import matplotlib.animation

# from pylab import *

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

        # graphs
        self.graph_data = []
        self.epsilon_graph_init = False

    '''
        Q-LEARN
    '''

    def reduce(self, obs):
        ratios = [(obs[i] + abs(self.lower_bounds[i])) / (abs(self.upper_bounds[i]) - self.lower_bounds[i]) for i in range(len(obs))]
        reduced_obs = [int(round((self.buckets[i]) - 1) * ratios[i]) for i in range(len(obs))]
        # reduced_obs = [min(self.buckets[i] - 1, max(0, reduced_obs[i])) for i in range(len(obs)) ]
        return tuple(reduced_obs)

    def choose_action(self, state, epsilon):
        if np.random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state])

    # exploration rate (discount factor)
    def calc_epsilon(self, time):
        # linear decay w.r.t. time
        epsilon = 1 - time/100
        return max(self.min_epsilon, epsilon)

        # logarithmic decay w.r.t. time
        if time == 0:
            return 1
        else:
            epsilon = 1 - math.log10(time)/2
            print('epsilon', epsilon)
            return max(self.min_epsilon, epsilon)

    # learning rate
    def calc_alpha(self, time):
        # linear decay w.r.t. time
        # alpha = 1 - time/100
        # return max(self.min_alpha, alpha)
        if time == 0:
            return 1
        else:
            alpha = 1 - math.log10(time)/2
            print('alpha', alpha)
            return max(self.min_alpha, alpha)


    def update_q(self, old_state, action, reward, new_state, alpha):
        self.q_table[old_state][action] += alpha * (reward + self.gamma * np.max(self.q_table[new_state]) - self.q_table[old_state][action])

    '''
        SIMULATION
    '''

    def run(self):
        plt.ion()
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()
        x, y, z, w = [],[],[],[]
        ax.set_ylim(0, 1)
        ax2.set_ylim(0, 1)
        ax3.set_ylim(0, 300)
        plt.draw()
        for i in range(self.num_episodes):
            epsilon = self.calc_epsilon(i)
            alpha = self.calc_alpha(i)
            count = 0
            total_reward = 0
            current_state = self.reduce(self.env.reset())
            done = False

            while not done:
                count += 1
                sleep(0.01)
                self.env.render()
                action = self.choose_action(current_state, epsilon)
                observation, reward, done, info = self.env.step(action)
                new_state = self.reduce(observation)
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                if done:
                    # self.graph_data.append({
                    #     'epsilon': epsilon,
                    #     'alpha': alpha,
                    #     'success': count
                    # })
                    x.append(i)
                    y.append(epsilon)
                    z.append(alpha)
                    w.append(count)
                    ax.plot(x, y)
                    ax2.plot(x, z)
                    ax3.plot(x, w)
                    plt.pause(0.1)

                    # self.update_charts()
                    print('failed after {0} samples'.format(count))
                    break


    '''
        GRAPHING
    '''

    def update_charts(self):
        if len(self.graph_data) > 1:
            self.update_epsilon_chart()
            # self.update_alpha_chart(self)
            # self.success_chart(self)

    def update_epsilon_chart(self):
        epsilon_data = []
        for data in self.graph_data[:]:
            epsilon_data.append(data['epsilon'])

        # data = [go.Scatter(
        #     x = np.linspace(0, 1, len(self.graph_data)),
        #     y = epsilon_data
        # )]

        # if self.epsilon_graph_init:
        #     print('here')
        #     py.iplot(data, filename='epsilon_chart.html', fileopt='extend')
        # else:
        #     self.epsilon_graph_init = True
        #     py.iplot(data, filename='epsilon_chart.html')
        # py.plot(data, filename='epsilon_chart', fileopt='extend')
        print('x axis', list(range(0, len(epsilon_data))), type(list(range(0, len(epsilon_data)))))
        plt.plot([1.6, 2.7])

    # def update_alpha_chart(self):
    #
    # def success_chart(self):

if __name__ == '__main__':
    options = {
        'num_episodes': 100,
        'num_win_ticks': 200,
        'num_theta_buckets': 5,
        'num_theta_prime_buckets': 5,
        'buckets': (1, 1, 6, 12),
        'min_epsilon': 0.1,
        'min_alpha': 0.1,
        'gamma': 1
    }

    learner = CartPoleLearner(options)
    learner.run()
