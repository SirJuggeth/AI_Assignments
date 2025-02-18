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

__version__ = "1.0.0"  # Version tracking

# =============================================================================
# Imports
# =============================================================================

import time
from typing import Any, Literal, SupportsFloat

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from click import FLOAT
from gymnasium import Env
from matplotlib.axes import Axes
from numpy.typing import NDArray
from pyglet.window import key

# =============================================================================
# Global Constants and Configurations
# =============================================================================
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
epsilon = 1

BURN_IN = 1  # When you start to reduce epsiolon.
EPSILON_END = 10000
EPSILON_REDUCE = 0.0001

env: Env[Any, Any] = gym.make("CartPole-v1")
env.reset()
env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

NPFloatArray = NDArray[np.float64]  # Type alias for clarity

# Elcploration vs. Exploitation param
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration prob at start
min_epsilon = 0.01  # Min exploration prob
decay_rate = 0.001  # Exponential decay for exploration


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
    action: int = None

    if k == key.LEFT:
        action = 0
    if k == key.RIGHT:
        action = 1

    return action


def create_bins(num_bins_per_obs: int = 10) -> NDArray[NPFloatArray]:
    """
    This function divides a span into discrete, equally sized portions.

    Args:
        num_bins_per_obs

    Returns:
        bins_cart_position: NDArray[np.float64]
    """
    # print(create_bins())

    bins_cart_position: NPFloatArray = np.linspace(
        -4.8, 4.8, num_bins_per_obs
    )  # dividing x-axis

    bins_cart_vel: NPFloatArray = np.linspace(
        5, 5, num_bins_per_obs
    )  # cart velocity is measured in m/s?

    bins_pole_angle: NPFloatArray = np.linspace(
        -0.418, 0.418, num_bins_per_obs
    )  # pole angle is measured in radians

    bins_pole_angular_vel: NPFloatArray = np.linspace(
        5, 5, num_bins_per_obs
    )  # pole angular velocity is measured in m/s?

    bins: NDArray[NPFloatArray] = np.array(
        [
            bins_cart_position,
            bins_cart_vel,
            bins_pole_angle,
            bins_pole_angular_vel,
        ]
    )
    print(bins)
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

    for i, obs in enumerate(observations):  #  if it didn't have i as part
        # of the statement it wouldn't have needed the enumeration.

        import pdb  # for troubleshooting

        pdb.set_trace()

        # find which bin obs fits into and puts it there.
        discretized_obs = np.digitize(obs, bins[i])
        binned_observations.append(discretized_obs)

    return tuple(binned_observations)  # Important for later indexing


def epsilon_greedy_action_selction(epsilon, q_table, discrete_state) -> int:
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
    rand_num: float = np.random.random()

    # EXPLOITATION, USE BEST Q(s, a) value
    if rand_num > epsilon:
        action: np.float64 = np.argmax((q_table[discrete_state]))

    # EXPLORATION, USE A RANDOM ACTION
    else:
        # Return a random action 0, 1, 2, 3
        action: Any = np.random.randint(0, env.action_space.n)

    return action


def compute_next_x_al(old_q_val, reward, next_optimaal_q_val) -> None:
    """
    Short desc of func.

    More detailed desc of func

    Args:
        param1: (int): Description.
        param2: (str): Description.

    Returns:
        None: Description of return value.
    """
    pass  # Placeholder for future code

    return None


def compute_next_q_val(
    old_q_val: int, reward: int, next_optimal_q_val: int
) -> None:
    """
    Short desc of func.

    More detailed desc of func

    Args:
        old_q_val, reward, next_optimal_q_val: (int): Description.
        reward: (str): Description.

    Returns:
        None: Description of return value.
    """
    return old_q_val + ALPHA * (reward + GAMMA * next_optimal_q_val)


def reduce_epsilon(epsilon: int, epoch: int) -> int:
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
        epsilon -= EPSILON_REDUCE

    return epsilon


def fail(terminated: int, truncated: int, points: int, reward: int) -> int:
    """
    Short desc of func.

    More detailed desc of func

    Args:
        terminated: (int): Description.
        truncated: (int): Description.

    Returns:
        int: Description of return value.
    """
    if (terminated | truncated) and points < 150:
        reward = -200

    return reward


# =============================================================================
# Main Execution Block
# =============================================================================
def main() -> None:
    """
    Main script execution
    """
    observations = env.reset()
    rewards = 0
    num_bins = 10
    bins: NDArray[NDArray[np.float64]] = create_bins(num_bins)

    mapped_observation: tuple = discretize_observation(observations, bins)
    # print(mapped_observation)
    # starting position is (5, 5, 5, 5)

    q_table_shape = (num_bins, num_bins, num_bins, num_bins, env.action_space.n)
    q_table = np.zeroes(q_table_shape)

    # Innitial trial code
    # for _ in range(1000):
    #     env.render()
    #     # get the reward and the done flag
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     rewards += 1
    #     if terminated | truncated:
    #         print(f"you got {rewards} points!")
    #         break
    #     time.sleep(0.5)
    # env.close()

    ##############################################
    ### VISUALIZATION OF TRAINING PROGRESS ######
    #############################################
    log_interval = (
        500  # How often do we update the plot? (Just for performance reasons)
    )
    render_interval = 2000  # How often to render the game during training
    # (If you want to watch your model learning)
    ### Here we set up the routine for the live plotting of the achieved points
    fig = plt.figure()
    ax: Axes = fig.add_subplot(111)
    plt.ion()
    fig.canvas.draw()
    ##############################################
    #############################################

    points_log: list = []
    mean_points_log: list = []
    epochs: list = []

    for epoch in range(EPOCHS):
        ## continuous state --> Discrete state

        initial_state: tuple[Any, dict[str, Any]] = (
            env.reset()
        )  # get the initial observation
        discretized_state: tuple = discretize_observation(
            initial_state, bins
        )  # map the observation to the bins

        terminated = truncated = (
            False  # to stop current run when cartpole falls down
        )
        points = 0

        # Track epochs for Plotting Visualization
        epochs.append(epoch)

        while (
            not terminated | truncated
        ):  # Perform current run as long as done is False (as long as the cartpole is up)
            # View how the cartpole is doing every render interval
            #         if epoch % render_interval == 0:
            #             env.render()

            action: int | Any = epsilon_greedy_action_selction(
                epsilon, q_table, discretized_state
            )  # Epsilon-Greedy Action Selection

            next_state, reward, terminated, truncated, info = env.step(action)

            reward = fail(terminated, truncated, points, reward)

            next_state_discretized = discretize_observation(next_state, bins)

            old_q_val = q_table[discretized_state + (acttion,)]
            next_optimal_q_val = np.max(q_table[next_state_discretized])

            next_q = compute_next_q_val(old_q_val, reward, next_optimal_q_val)
            q_table[discretized_state + (action,)] = next_q

            discretized_state = next_state_discretized
            point += 1

        espsilon = reduce_epsilon(epsilon, epoch)

        points_log.np.append(points)
        running_mean = round(np.mean(points_log[-30:]), 2)
        mean_points_log, append(running_mean)

        ################ Plot the points and running mean ##################
        if epoch % log_interval == 0:
            ax.clear()
            ax.scatter(epochs, points_log)
            ax.plot(epochs, points_log)
            ax.plot(
                epochs, mean_points_log, label=f"Running Mean: {running_mean}"
            )
            plt.legend()
            flg.canvas.draw()
        ######################################################################
        env.close()

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
