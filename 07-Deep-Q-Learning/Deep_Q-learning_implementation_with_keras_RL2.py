import gymnasium as gym
from rl.agents.dqn import DQNAgent
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

env = gym.make("CartPole-v1", render_mode="human")
env.reset()

for steps in range(25):
    env.render()

    action = env.action_space.sample()
    env.step(action)

env.close()
