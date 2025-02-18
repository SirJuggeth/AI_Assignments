# =============================================================================
# Module Information
# =============================================================================

"""
Module Name: Discrete_Q-learning_implementation_ForzenLake
Version: 1.0.0
Author: Justin J. Harrott
Email: Sir.Juggeth@Gmail.com
Date: 2025-02-12

Description:
    While playing the Classic Control Cart Pole game, our created agent will populate
    a table with Q-values, representing the likelihood of obtaining a reward for making
    a specific action while in a certain state.

    The Cart Pole game's intention is to have the pole vertically ballanced on top-centre of
    a cart that moves left and right.

    There are 4 observations:
    * Cart Position (-4.8 to 4.8)
    * Cart Velocity $(-\infty \text{ to } \infty)$
    * Pole Angle (-0.418 to 0.418) rad or (-24 to 24) degrees
    * Pole Angular Velocity $(-\infty \text{ to } \infty)$

    and two actions:
    * 0 - Move to the left
    * 1 - Move to the right

    The agent is rewarded ...

    * Overly elaborate and random notes have been made and functions have been left lengthy
    # to provide clarity for learning and reference purposes.


    ===========================================
    Q-learning update rule:
    Q(s_t, a_t) = Q(s_t, a_t) + alpha * (
    r_t+1 + gamma * max_a Q(s_t+1, a) - Q(s_t, a_t)
    )


    Where:
    - Q(s_t, a_t): Q-value for state s_t and action a_t
    - alpha: Learning rate (0 < alpha <= 1)
    - r_t+1: Reward received after taking action a_t in state s_t
    - gamma: Discount factor (0 <= gamma <= 1)
    - max_a Q(s_t+1, a): Maximum Q-value for the next state s_t+1
    ===========================================


Dependencies:
    - dependency1
    - dependency2

Usage:
    ```python
    import module_name
    example_usage
    ```
"""

# ===========================================
# Imports
# ===========================================

import time

import ale_py
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.registration import register
from IPython.display import clear_output

"""
===========================================
Global Constants and Configurations
===========================================
"""
EPOCHS = 2000  # Same as EPISODES - how many times the agent plays the game.

ALPHA = 0.8  # Same as LEARNING RATE - should be close to, but not equal to 1.
# Set too low, and learning takes too long. Set too high, and it may never
# converge due to unstable, overly reactive Q-values, causing excessive
# fluctuations and loss of long-term trends by overwriting useful data.

GAMMA = 0.95  # Same as DISCOUNT RATE - should be close to, but not equal to 1.
# A high discount rate (close to 1) prioritizes future rewards but slows
# convergence and can cause instability. A low rate (close to 0) makes it
# short-sighted, ignoring long-term rewards and converging to suboptimal
# policies. The best choice depends on the environment. Using an exponential
# decay function is a good approach, but a linear decrease is also common.

# Instantiate gymnasium environment.
env = gym.make(
    "FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="ansi"
)

# Instantiate Q-table.
state_size = env.observation_space.n  # In a 4x4 grid, each position (or cell) in the grid is a state. Since there are 4 rows
# and 4 columns, the total number of states is: 4 * 4, So, there are 16 states in a 4x4 grid.
action_size = env.action_space.n  # (actions are 0-3: left, down, right, up ∴ size is 4)
# Rows are States, columns are Actions.
q_table = np.zeros([state_size, action_size])  # creates 16x4 array of 0s
"""
Each state (square on the game board) has its own row in the q_table representating the reward vulues for each action (to move in a
different direction), they are sahowing as columns in the q_table. The Q-values in the table tell you the expected reward for each
action taken from each square.
"""
epsilon = 1.0  # Exploration vs exploitation parameter. Usually starts at 1 (exploration) and uses a function to decrease
# it as epochs are run.
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.001

rewards = []

print(matplotlib.get_backend())
fig = plt.figure()
ax = fig.add_subplot(111)
plt.ion()  # Enable interactive plot mode
plt.show(block=False)
fig.canvas.draw()
epoch_plot_tracker = []
total_reward_plot_tracker = []


"""
===========================================
Functions and Utility Definitions
===========================================
"""


def epsilon_greedy_action_selection(epsilon, q_table, discrete_state):
    random_number = np.random.random()  # random number between 0-0.1 (non-inclusive)

    # EXPLOITATION (choose the action that maximizes Q) as epsilon gets smaller and smaller, the function will get greedier and greedier
    # going with what it has found has a better reward
    if random_number > epsilon:
        state_row = q_table[discrete_state, :]
        action = np.argmax(
            state_row
        )  # grabs the index (position in row, 0-3) of the action with the highest potential reward

    # EXPLORATION (choose a random action)
    else:
        action = env.action_space.sample()

    return action


def compute_next_q_value(old_q_value, reward, next_optimal_q_value):
    """
    ===========================================
    Q-learning update rule:
    Q(s_t, a_t) = Q(s_t, a_t) + alpha * (r_t+1 + gamma * max_a Q(s_t+1, a) - Q(s_t, a_t))

    Where:
    - Q(s_t, a_t): Q-value for state s_t and action a_t
    - alpha: Learning rate (0 < alpha <= 1)
    - r_t+1: Reward received after taking action a_t in state s_t
    - gamma: Discount factor (0 <= gamma <= 1)
    - max_a Q(s_t+1, a): Maximum Q-value for the next state s_t+1
    ===========================================

    Q-value is the likeliness of getting to the reward by using this action.
    """
    return old_q_value + ALPHA * (reward + GAMMA * next_optimal_q_value - old_q_value)


def reduce_epsilon(epsilon, epoch):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(
        -decay_rate * epoch
    )  # exponential decay


"""
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
"""


"""
===========================================
Main Logic / Execution Block
===========================================
"""


def main():
    try:
        register(
            id="FrozenLake-v1",  # Unique environment ID
            entry_point="gymnasium.envs.toy_text:FrozenLakeEnv",  # Path to the environment class
            max_episode_steps=100,  # Optional: max steps per episode
            # reward_threshold=0.78,  # Optional: reward threshold for success. Optimum for FrozenLake is 0.8196
        )
    except:
        print("Registration name already taken")

    """
    # basic script to see the game work
    # Initialise the environment
    env.reset()

    # Reset the environment to generate the first observation OBSERVATION,
    # which is a sample of data from the complete set of STATE data that is a complete
    # description of the environment.
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        # this is where you would insert your policy

        env.render()
        action = env.action_space.sample() #simple_agent(observation)
        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)
        # time.sleep(0.3)
        clear_output(wait=True)

        # plt.draw()  # Update plot
        # plt.pause(0.5)  # Pause to allow visualization

        # If the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()

    env.close()
    """

    # training_loop
    log_interval = 200

    for episode in range(EPOCHS):
        global epsilon
        state, state_info = (
            env.reset()
        )  # returns tuple[ObsType, dict[str, Any]] meaning
        # [Observation (state), info {keyword, data}] - prob means probability
        terminated = truncated = False
        total_rewards = 0

        while not terminated | truncated:
            # ACTION - generate a new action
            action = epsilon_greedy_action_selection(
                epsilon, q_table, state
            )  # will it be a random action
            # or prioritize what it knows is a better choice

            # state, reward... env.step - implement action and see what happens
            new_state, reward, terminated, truncated, info = env.step(action)
            # new_state is a complete description of the next step taking action in the environment.
            print(new_state, reward, terminated, truncated, info, epsilon)

            # OLD (current) Q-value table Q(s_t, a_t)
            old_q_value = q_table[state, action]

            # get next optimal Q-value (max Q for this state) Q(s_t+1, a_t+1)
            next_optimal_q_value = np.max(q_table[new_state, :])

            # compute the next Q-value
            next_q = compute_next_q_value(old_q_value, reward, next_optimal_q_value)

            # Update the Q-value table
            q_table[state, action] = next_q

            # track rewards
            total_rewards = total_rewards + reward

            # new state is now the current state
            state = new_state

        # Agent finished a round of the game
        episode += 1
        epsilon = reduce_epsilon(epsilon, episode)
        rewards.append(total_rewards)

        total_reward_plot_tracker.append(np.sum(rewards))
        epoch_plot_tracker.append(episode)

        if episode % log_interval == 0:
            print(np.sum(rewards))
            ax.clear()
            ax.plot(epoch_plot_tracker, total_reward_plot_tracker)
            fig.canvas.draw()
            plt.pause(0.1)  # Give time for the UI to update
            time.sleep(1)

        # if completed episodes, show found winning strategy
        if episode == EPOCHS:
            env.reset()

            for steps in range(50):
                # env.render()
                print("Rendered", env.render())
                action = np.argmax(q_table[state, :])
                state, reward, terminated, truncated, info = env.step(action)

                plt.pause(0.1)  # Give time for the UI to update
                time.sleep(1)
                clear_output(wait=True)

                if terminated:
                    print("Rendered", env.render())
                    print("ALL DONE!")
                    break

    env.close()


if __name__ == "__main__":
    main()
