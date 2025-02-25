import gymnasium as gym
import ale_py

def simple_agent(observation):
    # OBSERVATION
    position, velocity = observation

    # When to go right -->
    if -0.1 < position < 0.4:
        action = 2

    # When to go left <--
    elif velocity < 0 and position < -0.2:
        action = 0

    # When to do nothing
    else:
        action = 1
    
    return action

# Initialise the environment
env = gym.make("MountainCar-v0", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = simple_agent(observation)
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()

