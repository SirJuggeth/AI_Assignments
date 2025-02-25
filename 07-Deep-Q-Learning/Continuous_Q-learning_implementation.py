# =============================================================================
# Module Information
# =============================================================================

"""
Module Name: Continuous_Q-learning_implementation
Version: 1.0.0
Author: Justin J. Harrott
Email: Sir.Juggeth@Gmail.com
Date: 2025-02-12

Description:
    While playing the Classic Control Cart Pole game, our created agent will
    populate a table with Q-values, representing the likelihood of obtaining a
    reward for making a specific action while in a certain state.

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
    Q-learning update rule (Bellman equation):
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

__version__ = "1.0.0"  # Version tracking

# =============================================================================
# Imports
# =============================================================================

import pdb
import sys
import time
import warnings
from typing import Any, Literal, SupportsFloat

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pyglet.window import key

# =============================================================================
# Global Constants and Configurations
# =============================================================================
EPOCHS = 20000  # Same as EPISODES - how many times the agent plays the game.

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

BURN_IN = 1  # When you start to reduce epsiolon.
EPSILON_END = 15000  # Epoch when to stop reducing epsilon
EPSILON_REDUCE = 0.0001

# Elcploration vs. Exploitation param
max_epsilon = 1.0  # Exploration prob at start
min_epsilon = 0.01  # Min exploration prob
decay_rate = 0.001  # Exponential decay for exploration

# Convert warnings to exceptions
# warnings.simplefilter("error", Warning) # converts warnings into errors.

env: Env[Any, Any] = gym.make("CartPole-v1", render_mode="rgb_array")
env.reset()
env = gym.wrappers.TimeLimit(env, max_episode_steps=500)

NPFloatArray = NDArray[np.float64]  # Type alias for clarity


# =============================================================================
# Classes and Functions
# =============================================================================
def key_press(k) -> None:
    """
    This function gets the key press for gym

    Args:
        None

    Returns:
        action
    """
    action = None

    if k == key.LEFT:
        action = 0
    if k == key.RIGHT:
        action = 1

    return action


def create_bins(num_bins_per_obs) -> NDArray[NPFloatArray]:
    """
    This function creeates empty bins that divide a span into discrete, equally
    sized portions. There will be bins for cart position, cart velocity,
    pole angle and pole angular velocity.

    Args:
        num_bins_per_obs: int

    Returns:
        bins: NDArray[np.float64]
    """
    # We've chosen 10 bins per span, but you can choose more or less. Choosing
    # less would cause the model to predict less meaningful information, but
    # there comes a point where having more bins does not provide any benefit.

    bins_cart_position: NPFloatArray = np.linspace(
        -4.8, 4.8, num_bins_per_obs
    )  # dividing x-axis

    bins_cart_vel: NPFloatArray = np.linspace(
        -5, 5, num_bins_per_obs
    )  # cart velocity is measured in m/s?
    # -inf, inf are the possible extremes, but that doesn't make sense in
    # this scenario. You can play to see what makes sense, but we will
    # assume -5, 5 is reasonable.

    bins_pole_angle: NPFloatArray = np.linspace(
        -0.418, 0.418, num_bins_per_obs
    )  # pole angle is measured in radians, but couldbe done in degreed also
    # -24 deg to 24 deg

    bins_pole_angular_vel: NPFloatArray = np.linspace(
        -5, 5, num_bins_per_obs
    )  # pole angular velocity is measured in m/s?

    bins: NDArray[NPFloatArray] = np.array(
        [
            bins_cart_position,
            bins_cart_vel,
            bins_pole_angle,
            bins_pole_angular_vel,
        ]
    )
    # print(bins)
    return bins


def discretize_observation(observations, bins) -> None:
    """
    Takes the given observations and puts them in their appropriate bins.

    Observations for cart position, cart velocity, pole angle,
    and pole angular velocity are given with a collection of bins representing
    the quantized scales for the different observation types.

    Args:
        observations (NDArray[np.float64]): numpy array of floats representing
        for cart position, cart velocity, pole angle,
        and pole angular velocity.

        bins (NPArray[NDArray[np.float64]]): 4x4 array, made up of the
        quantized scales for the different position and vel observation types.

    Returns:
        tuple[int, ...]: Observations divided into their appropriate bins.
    """
    binned_observations: list = []

    if type(observations) is tuple and len(observations) > 1:
        observations = observations[0]

    for i, obs in enumerate(observations):  #  if it didn't have i as part
        # of the statement it wouldn't have needed the enumeration.

        # import pdb  # for troubleshooting

        # pdb.set_trace()

        # find which bin obs fits into and puts it there.
        discretized_obs = np.digitize(obs, bins[i])
        binned_observations.append(discretized_obs)

    return tuple(binned_observations)  # Important for later indexing when
    # creating the Q_table


def epsilon_greedy_action_selction(epsilon, q_table, discrete_state, weights) -> int:
    """
    Returns an action for the agent. Note how it uses a random number to decide on
    exploration versus explotation trade-off.

    More detailed desc of func

    Args:
        epsilon): Description.
        param2: (str): Description.

    Returns:
        None: Description of return value.
    """

    rand_num = np.random.random()

    # EXPLOITATION, USE BEST Q(s, a) value
    if rand_num > epsilon:
        action = np.argmax(q_table[discrete_state])

    # EXPLORATION, USE A RANDOM ACTION
    else:
        # Return a random action 0, 1, 2, 3
        action = np.random.randint(0, env.action_space.n)

    return action


def compute_next_q_val(current_q_val, reward, next_optimal_q_val, i):
    """
    Short desc of func.

    More detailed desc of func

    Args:
        current_q_val, reward, next_optimal_q_val: (int): Description.
        reward: (str): Description.

    Returns:
        None: Description of return value.
    """
    """
    try:
        print("next Q epoch: ", i)
        q = current_q_val + ALPHA * (
            reward + GAMMA * next_optimal_q_val - current_q_val
        )

    except Warning as e:
        # Log or print variables if you want a quick inspection before pausing
        print(f"Overflow encountered at iteration {i} because of {e}")
        print(
            f"current_q_val: {current_q_val}, reward: {reward}, next_optimal_q_val: {next_optimal_q_val}"
        )

        # Pause the program here without breaking the loop
        pdb.set_trace()
"""

    return current_q_val + ALPHA * (
        reward + GAMMA * next_optimal_q_val - current_q_val
    )


def determine_weights(discretized_state):
    cart_pos = discretized_state[1]  # Alias, for clarity
    cart_vel = discretized_state[0]  # Alias, for clarity
    pole_pos = discretized_state[2]  # Alias, for clarity
    pole_ang_vel = discretized_state[3]  # Alias, for clarity

    if pole_pos and pole_ang_vel < 0:
        weights = [0.0, 0.25]

    if pole_pos and pole_ang_vel > 0:
        weights = [0.25, 0.0]

    return weights


def reduce_epsilon(epsilon, epoch, weights):
    """
    Linear reduction of epsilon, with a burn in and hard stop point.

    More detailed desc of func

    Args:
        epsilon: (int): Description.
        epoch: (int): Description.

    Returns:
        int: Description of return value.
    """
    if BURN_IN <= epoch <= EPSILON_END:
        epsilon -= EPSILON_REDUCE  # Linear reduction of epsilon until number of
        # epochs reaches EPSILON_END. BURN_IN is when exploitation begins.
        # It could be set to start at a later epoch.

    return epsilon


def fail(terminated, truncated, points, reward) -> int:
    """
    Punishes agent for not getting enough points during an epoch.

    You have the ability to give rewards or punishment based on certain param,
    like angular velocity.

    Args:
        terminated: (int): Description.
        truncated: (int): Description.

    Returns:
        int: Description of return value.
    """
    if (terminated | truncated) and points < 150:
        print("FAILLED")
        reward = -200

    return reward


def crash_handler(type, value, traceback):
    print("\n🔥 Crash detected! Entering post-mortem debugging...")
    pdb.post_mortem(traceback)  # Drop into debugger


# Set post-mortem debugging as the default handler
sys.excepthook = crash_handler


# =============================================================================
# Main Execution Block
# =============================================================================
def main() -> None:
    """
    Main script execution
    """
    file = open("Q-tables.txt", "a")

    observations = env.reset()
    print("1st obs: ", observations)

    # rewards = 0
    points = 0
    num_bins = 10
    bins = create_bins(num_bins)

    points_log: list = []
    mean_points_log: list = []
    epochs: list = []

    epsilon = 1.0  # Exploration rate

    q_table_shape = (num_bins, num_bins, num_bins, num_bins, env.action_space.n)
    q_table = np.zeros(q_table_shape)
    # print("Q-table: ", q_table)

    # How often do we update the plot? (Just for performance reasons)
    log_interval = 500

    render_interval = 200  # How often to render the game during training
    # (If you want to watch your model learning, see the game being played)

    ##############################################
    ### VISUALIZATION OF TRAINING PROGRESS ######
    #############################################
    ### Here we set up the routine for the live plotting of the achieved points
    fig = plt.figure()
    ax: Axes = fig.add_subplot(111)
    plt.ion()
    plt.show()
    fig.canvas.draw()
    ##############################################
    #############################################

    for epoch in range(EPOCHS):
        ## continuous state --> Discrete state

        # get the initial observation
        initial_state = env.reset()

        # map the observation to the bins
        discretized_state = discretize_observation(initial_state, bins)

        determine_weights(discretized_state)

        # initialize var that will stop current run when cartpole falls down
        terminated = truncated = False

        points = 0  # reset points counter

        # Track epochs for Plotting Visualization
        epochs.append(epoch)

        # Play the game
        while not terminated | truncated:
            # Perform current run as long as terminated | truncated is False
            # (as long as the cartpole is up)

            action = epsilon_greedy_action_selction(
                epsilon, q_table, discrete_state I need to figure out descrete vs discretized
            )

            next_state, reward, terminated, truncated, info = env.step(action)

            reward = fail(terminated, truncated, points, reward)

            next_state_discretized = discretize_observation(next_state, bins)

            current_q_val = q_table[discretized_state + (action,)]
            next_optimal_q_val = np.max(q_table[next_state_discretized])

            next_q = compute_next_q_val(
                current_q_val, reward, next_optimal_q_val, epoch
            )
            q_table[discretized_state + (action,)] = next_q

            discretized_state = next_state_discretized
            points += 1  # +1 point for each action taken

        epsilon = reduce_epsilon(epsilon, epoch, discretized_state)

        print("points: ", points)
        points_log.append(points)
        # Calculate mean of last 30 games, rounded to 2 decimal points
        # for the purpose of visualization
        running_mean = round(np.mean(points_log[-30:]), 2)
        mean_points_log.append(running_mean)

        ################ Plot the points and running mean ##################
        if epoch % log_interval == 0:
            print("log interval, epoch: ", epoch)
            ax.clear()
            ax.scatter(epochs, points_log)
            ax.plot(epochs, points_log)
            ax.plot(
                epochs, mean_points_log, label=f"Running Mean: {running_mean}"
            )
            # plt.pause(0.01)  # Give time for the UI to update
            # time.sleep(1)
            # plt.legend()
            fig.canvas.draw()

            # ####log q_table for troublehooting
            # # Convert the updated Q-table to a string, maintaining its structure
            # q_table_string = np.array2string(
            #     q_table, separator=",", threshold=np.inf
            # )
            # # Write the Q-table snapshot to the file with newline separation
            # file.write(
            #     q_table_string + "\n\n"
            # )  # Double newline for clearer separation
        ######################################################################
    env.close()
    file.close()

    # action
    #     # perform action and get next state

    # reward: SupportsFloat | Literal[-200] = fail(
    #     terminated, truncated, points, reward
    # )  # Check if reward or fail state

    # next_state_discretized = discretize_observation(
    #     next_state, bins
    # )  # map the next observation to the bins

    return None


if __name__ == "__main__":
    main()
