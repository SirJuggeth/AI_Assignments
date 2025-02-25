import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.optimizers import Adam

EPOCHS = 50
EPSILON_DECAY_RATE = 0.995

BATCH_SIZE = 32

LEARNING_RATE = 0.001  # Different from ALPHA. This is for optimization of ANN.
GAMMA = 0.95

env = gym.make("CartPole-v1", render_mode=None)
env.reset()


"""
# # Basic environment render test
# env = gym.make("CartPole-v1", render_mode="human")
# env.reset()

# for step in range(100):
#     env.render()
#     env.step(env.action_space.sample())  # perform random action

# env.close()
"""


def epsilon_greedy_action_selection(model, epsilon, obs):
    # print(f"Observation shape: {np.array(obs).shape}")
    obs = np.array(obs).reshape(1, 1, 4)
    # print(f"Observation re-shape: {np.array(obs).shape}")

    if np.random.random() > epsilon:
        prediction = model.predict(obs, verbose=0)  # return ex. [0.4, 0.6]
        action = np.argmax(prediction)  # return index of highest ^
    else:
        # Random int from 0 to (non-inclusive)number of actions in space
        # (2 in CartPole)
        action = np.random.randint(0, env.action_space.n)

    return action


def replay(replay_buffer, batch_size, model, target_model):
    # Grab samples once replay_buffer has reached batch size threshold.
    # Can't proceed unless batch size limit has been reached.
    if len(replay_buffer) < batch_size:
        return None

    # Crate batch_size sized list of random samples taken from buffer.
    samples = random.sample(replay_buffer, batch_size)

    # Initialize list to store predicted targets by the target network.
    target_batch = []

    # "Zip" together samples from replay buffer.
    # Sample  = (state, action, reward, next state, terminated, truncated)
    # Zipping together iterates through samples and puts state with state,
    # action with action, ... etc. from each sample
    zipped_samples = list(zip(*(samples)))

    (
        observations,
        actions,
        rewards,
        next_observations,
        terminateds,
        truncateds,
    ) = zipped_samples

    # FOR DEBUGGING
    # print(f"Next Observations shape: {np.array(next_observations).shape}")
    # print(f"Observations shape: {np.array(observations).shape}")

    # Predict targets for all states from the samples.
    # They are estimates used to update the Q-vals.
    # 𝑦 = 𝑟 + 𝛾 max 𝑄_target(𝑠′,𝑎′)
    # Where:
    # 𝑦 = predicted target (TD target)
    # 𝑟 = reward for the action
    # 𝛾 = discount factor
    # max 𝑄_target(𝑠′,𝑎′) = max Q-value of the next state from the target network
    targets = target_model.predict(np.array(observations), verbose=0)

    # Predict Q-vals for all new_states from the samples.
    q_values = model.predict(np.array(next_observations), verbose=0)

    for i in range(batch_size):
        q_value = max(q_values[i][0])
        target = targets[i].copy()

        if terminateds[i] | truncateds[i]:
            target[0][actions[i]] = rewards[i]
        else:
            target[0][actions[i]] = rewards[i] + q_value * GAMMA

        target_batch.append(target)

    # print("\nModel Fit\n")
    # Update fitting the model based on the states and updated targets of 1 epoch.
    model.fit(
        np.array(observations),
        np.array(target_batch),
        epochs=1,
        verbose=0,
    )

    return None


def update_model_handler(epoch, update_target_model, model, target_model):
    if epoch > 0 and epoch % update_target_model == 0:
        target_model.set_weights(model.get_weights())

    return None


def show_work(model, best_so_far):
    print("\nProof of how well model has been trained.\n")

    env = gym.make("CartPole-v1", render_mode="human")

    obs, info = env.reset()

    points = 0

    for epoch in range(best_so_far + 10):
        obs = obs.reshape(1, 1, 4)
        action = np.argmax(model.predict(obs, verbose=0))
        obs, reward, term, trunc, info = env.step(action)
        env.render()
        time.sleep(0.05)
        points += 1

        if term | trunc:
            break

    print(f"\nEpoch {epoch}: POINTS: {points} eps: Best so far: {best_so_far}")
    return None


def main():
    print("TF ver: ", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("TF config GPUs: ", tf.config.list_physical_devices("GPU"))
    print(
        "Num GPUs Available: ",
        len(tf.config.experimental.list_physical_devices("GPU")),
    )
    input("Press Enter to continue...")
    ### Create model ###
    # Define input (observations) and output (action) shape - If you had more
    # inputs than number of observations, you would have duplicate values
    #  - one observation for multiple inputs.
    # Number of layers and neurons can be played with.
    # CartPole has 4 observations: cart pos, cart vel, pole pos, pole ang_vel
    # Number of actions is 2: move left, move right

    num_obs = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = Sequential()

    # 16 inputs, 4 observations - each obs is going to 4 inputs
    model.add(Dense(16, input_shape=(1, num_obs)))  # specified input shape
    # as 1 dimensional array with 4 features (num_obs)
    model.add(Activation("relu"))

    model.add(Dense(32))
    model.add(Activation("relu"))

    # Output (Given as percent each action should be considered the right one)
    model.add(Dense(num_actions))
    model.add(Activation("linear"))

    model.summary()  # shows 690 trainable params
    # ((4 obs * 16 neurons (1st layer)) + 16 bias + (16 1st layer *
    # 32 2nd layer) + 32 bias + (32 2nd layer * 2 outputs) + 2 outputs)

    ### Create target network ###
    # Copy of the Q-network, updated less frequently,helping to reduce
    # instability caused by rapidly changing target values in the Q-network.
    target_model = clone_model(model)
    update_target_model = 10

    epsilon = 1.0

    ### Create replay buffer ###
    # Create deque with max length. If more than maxlen items are added (appended)
    # to deque, it will discard the first entry and append the new one on end.
    # replay buffer stores past experiences (state, action, reward, next state)
    # during training. It allows the agent to sample random mini-batches of
    # experiences to break correlations between consecutive experiences and
    # improve training stability.
    replay_buffer = deque(maxlen=20000)

    # Instantiate the optimizer with the desired learning rate
    optimizer = Adam(learning_rate=LEARNING_RATE)

    # Compile the model with the specified loss function and optimizer
    model.compile(loss="mse", optimizer=optimizer)

    # Initialize tracker for best score of points by keeping pole up.
    # Essentially just the reward.
    best_so_far = 0

    for epoch in range(EPOCHS):
        print("\nEpoch ", epoch, ":")

        obs, info = env.reset()  # returns tuple (obs_array, info_dict)
        obs = obs.reshape([1, 4])  # 1 dimensional array with 4 entries.
        terminated = truncated = False

        points = 0

        # Train model while it's playing the game.
        while not terminated | truncated:
            action = epsilon_greedy_action_selection(model, epsilon, obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = next_obs.reshape(
                [1, 4]
            )  # 1 dimensional array with 4 entries.

            replay_buffer.append(
                (obs, action, reward, next_obs, terminated, truncated)
            )

            obs = next_obs
            points += 1

            # Re-train model - update Q-vals and re-fit model based on them.
            replay(replay_buffer, BATCH_SIZE, model, target_model)

        epsilon *= EPSILON_DECAY_RATE

        update_model_handler(epoch, update_target_model, model, target_model)

        if points > best_so_far:
            best_so_far = points

        if epoch % 25 == 0:
            print(
                f"\nEpoch {epoch}: POINTS: {points} eps: {epsilon} Best so far: {best_so_far}"
            )

    show_work(model, best_so_far)

    env.close()

    return None


if __name__ == "__main__":
    main()
