from time import sleep
import gym

env = gym.make('HandManipulateBlock-v0')
print(env.__dict__)
print(env.action_space)
# env.render()
# env.HandBlockEnv._reset_sim()

for i in range(100):
    env.render()
    observation = env.reset()
    print(observation)
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    sleep(.1)
    if done:
        break
