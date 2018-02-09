import gym
import time
env = gym.make('CartPole-v0')

# print(gym.envs.registry.all())

for episode in range(20):
    observation = env.reset()
    done = False
    timestamp = 0

    while not done:
        timestamp = timestamp + 1
        env.render()
        print(observation)
        action = env.action_space.sample()
        print("action: {}".format(action))
        observation, reward, done, info = env.step(action)
        print("Reward: {}".format(reward))
        if done:
            print("Episode finished after {} timestamps".format(timestamp))
