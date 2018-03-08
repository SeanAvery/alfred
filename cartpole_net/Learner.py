import numpy as np

class QLearningAgent:
    def __init__(self, opts):
        # hyperparameters
        self.gamma = opts['gamma']
        self.epsilon = opts['epsilon']
        self.epsilon_decay = opts['epsilon_decay']
        self.epsilon_min = opts['epsilon_min']
        self.batch_size = opts['batch_size']

if __name__ == '__main__':
    options = {
        'gamma': 0.95,
        'epsilon': 1.0,
        'epsilon_decay': 0.995,
        'epsilon_min': 0.1,
        'batch_size': 32
    }
    learner = QLearningAgent(options)
