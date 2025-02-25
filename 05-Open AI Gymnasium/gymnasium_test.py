import gymnasium as gym
import ale_py
import matplotlib.pyplot as plt
import time

env_name = 'ALE/Breakout-v5'

# Initialise the environment
env = gym.make(env_name, render_mode="human")
env.reset()
'''
from gym.utils import play
play.play(env, zoom=3) #this is legacy, doesn't work in gymnasium
array = env.render(mode='rgb_array')
print(array)
'''


# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample() # randomly samples from action space (what kind of input the game can receive - up, down, left, right, etc) 
    #depending on the game, it could be an int between 0-17

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}, Truncated {truncated}")
    print(f"Info: {info}")

    time.sleep(0.05)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()