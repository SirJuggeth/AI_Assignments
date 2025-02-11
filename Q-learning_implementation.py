"""
===========================================
Imports
===========================================
"""
import gymnasium as gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from IPython.display import clear_output

from gymnasium.envs.registration import register



"""
===========================================
Global Constants and Configurations
===========================================
"""
EPOCHS = 2000 # same as EPISODES - how many times the agent plays the game.
ALPHA = 0.8 # same as LEARNING RATE - should be close to, but not equal to 1. Set too low and will take too long to learn, 
# set too high and it may never converge bacause it causes unstable, overly reactive Q-values, preventing convergence due 
# to excessive fluctuations and loss of long-term trends (it overwrites useful data that has already been learned).
GAMMA = 0.95 # same as DISCOUNT RATE - should be close to, but not equal to one. A high discount rate (close to 1) 
# makes the model prioritize future rewards, which can slow convergence and cause instability. A low rate (close to 0)
# makes it short-sighted, ignoring long-term rewards and converging to suboptimal policies. Depends on your environment 
# for which side you lean to. Good way is to use exponential decay function, but a linear decrease of a certain vlue is common.

# Instantiate gymnasium environment.
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="ansi")

# Instantiate Q-table.
state_size = env.observation_space.n #In a 4x4 grid, each position (or cell) in the grid is a state. Since there are 4 rows 
# and 4 columns, the total number of states is: 4 * 4, So, there are 16 states in a 4x4 grid.
action_size = env.action_space.n # (actions are 0-3: left, down, right, up ∴ size is 4)
# Rows are States, columns are Actions.
q_table = np.zeros([state_size, action_size]) # creates 16x4 array of 0s
'''
Each state (square on the game board) has its own row in the q_table representating the reward vulues for each action (to move in a 
different direction), they are sahowing as columns in the q_table. The Q-values in the table tell you the expected reward for each 
action taken from each square.
'''
epsilon = 1.0 # Exploration vs exploitation parameter. Usually starts at 1 (exploration) and uses a function to decrease 
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
    random_number = np.random.random() # random number between 0-0.1 (non-inclusive)
    
    # EXPLOITATION (choose the action that maximizes Q) as epsilon gets smaller and smaller, the function will get greedier and greedier
    # going with what it has found has a better reward
    if random_number > epsilon:
        state_row = q_table[discrete_state, :]
        action = np.argmax(state_row) # grabs the index (position in row, 0-3) of the action with the highest potential reward

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
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * epoch) # exponential decay

'''
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
'''



"""
===========================================
Main Logic / Execution Block
===========================================
"""
def main():
    try:
        register(
            id='FrozenLake-v1',  # Unique environment ID
            entry_point='gymnasium.envs.toy_text:FrozenLakeEnv',  # Path to the environment class
            max_episode_steps=100  # Optional: max steps per episode
            #reward_threshold=0.78,  # Optional: reward threshold for success. Optimum for FrozenLake is 0.8196
        )
    except:
        print('Registration name already taken')

    '''
    # basic script to see the game work
    # Initialise the environment
    env.reset()

    # Reset the environment to generate the first observation OBSERVATION is same as STATE
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
    '''

    # training_loop
    log_interval = 200

    for episode in range(EPOCHS):

        global epsilon 
        state, state_info = env.reset() # returns tuple[ObsType, dict[str, Any]] meaning 
        # [Observation (state), info {keyword, data}] - prob means probability
        terminated = truncated = False
        total_rewards = 0

        while not terminated | truncated:
            # ACTION - generate a new action
            action = epsilon_greedy_action_selection(epsilon, q_table, state) # will it be a random action 
            # or prioritize what it knows is a better choice
            
            # state, reward... env.step - implement action and see what happens
            new_state, reward, terminated, truncated, info = env.step(action) # new_state is same as observation
            print(new_state, reward, terminated, truncated, info, epsilon)

            # OLD (current) Q-value table Q(s_t, a_t)
            old_q_value = q_table[state, action]

            # get next optimal Q-value (max Q for this state) Q(s_t+1, a_t+1)
            next_optimal_q_value = np.max(q_table[new_state,:])

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
                action = np.argmax(q_table[state,:])
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