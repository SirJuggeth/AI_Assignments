from collections import deque

import ale_py  # is being accessed even though pylance says it isn't.
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.utils import play
from PIL import Image
from rl.agents.dqn import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.core import Processor
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow.keras.layers import (
    Activation,
    Convolution2D,
    Dense,
    Flatten,
    Permute,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

RENDER_MODES = [
    None,
    "human",
    "rgb_array",
    "ansi",
    "rgb_array_list",
    "human_pixels",
]
ENV_NAME = "ALE/Breakout-v5"

nb_actions = None
nb_obs = None
NB_STEPS = 1000000

WINDOW_LENGTH = 4
# *More complex games will use a larger window length (in frames)
IMG_SIZE = (84, 84)

CHECKPOINT_FILENAME = "DQN_checkpoints_Breakout.h5f"  # Unfinished weights.
WEIGHTS_FILENAME = "DQN_weights_Breakout.h5f"  # Fully trained weights.


class GymWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info, _ = self.env.step(action)
        return obs, reward, done, {}


class BreakOutProcessor(Processor):
    def process_observation(self, observation: np.ndarray):
        # Convert observation from numpy.ndarray from the sequntial_frame_buffer
        # to  PIL Image so it will work with the Python Imaging Library
        img = Image.fromarray(observation)

        # convert Breakout obs shape from (210, 160, 3)
        # 210 height x 160 width with 3 colour channels RGB
        # to (84,84) to reduce resolution for quicker processing.
        img = img.resize(IMG_SIZE)

        # Convert from 3 channel RGB colour to single channel grayscale
        # (L stands for luminescence)
        img = img.convert("L")

        # convert back to np.ndarray
        return np.array(img)


class ImageProcessor(Processor):
    def process_observation(self, observation):
        img = Image.fromarray(np.array(observation[0]))
        img = img.resize(IMG_SIZE)
        img = img.convert("L")  # Convert to grayscale (luminosity)
        img = np.array(img, dtype="uint8")  # Reduce bit depth to uint8

        print(
            "Processed observation shape before expanding:", img.shape
        )  # Should be (84, 84)

        img = np.expand_dims(
            img, axis=0
        )  # ✅ Add 1st dim so it stacks correctly later: (1, 84, 84)

        print(
            "Processed observation shape after expanding:", img.shape
        )  # Should be (1, 84, 84)

        # return img with optimized bit-length, so nums don't use too many bits,
        # like floats would.
        return img

        # Scale images - instead of going from 0.0-255.0, they go from 0.0-1.0.
        """    def process_state_batch(self, batch):
        print(f"batch type: {type(batch)}, batch length: {len(batch)}")
        if isinstance(batch, list):
            print(
                f"First element type: {type(batch[0])}, shape: {np.array(batch[0]).shape if isinstance(batch[0], np.ndarray) else 'N/A'}"
            )

        batch = np.array(
            batch, dtype="object"
        )  # Avoid immediate dtype enforcement
        print(f"Batch shape before conversion: {batch.shape}")

        try:
            batch = (
                np.stack(batch).astype("float32") / 255.0
            )  # Force uniform shape
        except Exception as e:
            print(f"Error during stacking: {e}")
            raise

        print(f"Processed batch shape: {batch.shape}")
        return batch"""

    def process_state_batch(self, batch):
        # Sanitize the batch to make sure all data is dtype uint8
        processed_batch = []
        for state in batch:
            state = np.array(
                state, dtype=np.uint8
            )  # Convert to uint8 (force dtype consistency)

            if state.shape == (84, 84):
                state = state.reshape(
                    1, 84, 84
                )  # Reshape single-frame states to (1, 84, 84)
            elif state.shape == (4, 84, 84):
                pass  # Already correctly stacked, leave as is
            else:
                raise ValueError(
                    f"Unexpected state shape: {state.shape}"
                )  # Raise error for unexpected shapes

            processed_batch.append(state)

        # Stack processed states into a batch (should now all be 3D arrays with shape (1, 84, 84))
        batch = np.stack(processed_batch)  # Stack into final array

        # We divide the observations by 255 to compress it into the intervall [0, 1].
        # This supports the training of the network
        # We perform this operation here to save memory.
        return batch.astype("float32") / 255.0

    def process_reward(self, reward):
        # This saves time on training.
        return np.clip(reward, -1.0, 1.0)


def create_model() -> Sequential:
    model = Sequential()

    # Define input shape for processing.
    # Input of 4 consecutive frames x images, shape of 84x84 pix (4, 84, 84).
    # Convolutional 2D layers expect input images x frame, shape  (Batch, 84, 84, 4).
    # We're going to add a permute layer at the beginningof our model modu
    input_shape = (
        WINDOW_LENGTH,
        IMG_SIZE[0],
        IMG_SIZE[1],
    )  # Tuple (None, 4, 84, 84) *None is the batch size

    # Permute(index, index, index, ...) - put indices of input in order of
    # how you want input to be permuted. Index 0 is placeholder for batch size.
    model.add(Permute((2, 3, 1), input_shape=input_shape))

    # 32 filters with 8x8 kernel size, convolution done a along 4 height
    # x 4 width strides. A 2D convolution filter is a small matrix (or kernel)
    # that slides over an image (or input data) to detect patterns, such as
    # edges, textures, or shapes.
    # keras.initializers defaults Convolution2D to 'glorot_uniform'. Another
    # common one is all zeros. These are weight initialization methods designed
    # to keep the variance of activations consistent across layers, preventing
    # gradients from vanishing or exploding.
    # he_normal (He initialization) is a weight initialization method optimized
    # for ReLU and its variants (LeakyReLU, ELU, etc.). It helps maintain stable
    # gradients by accounting for the non-linearity of ReLU, which only allows
    # positive activations.
    model.add(
        Convolution2D(
            32, (8, 8), strides=(4, 4), kernel_initializer="he_normal"
        )
    )
    model.add(Activation("relu"))

    model.add(
        Convolution2D(
            64, (4, 4), strides=(2, 2), kernel_initializer="he_normal"
        )
    )
    model.add(Activation("relu"))

    model.add(
        Convolution2D(
            64, (3, 3), strides=(1, 1), kernel_initializer="he_normal"
        )
    )
    model.add(Activation("relu"))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))

    # Output layer action ---> Q_(s,a) Q-values for each action of a given state.
    model.add(Dense(nb_actions))
    model.add(Activation("linear"))

    print(model.summary())

    return model


def set_agent_policy(model) -> DQNAgent:
    # Define replay buffer memory.
    mem = SequentialMemory(limit=NB_STEPS, window_length=WINDOW_LENGTH)

    processor = ImageProcessor()

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=NB_STEPS,  # Max number of actions per training step.
    )

    # Training interval is set to same as frame skip so that training only
    # happens when new info is gathered.
    # Clipped delta is a technique used in reinforcement learning to limit the
    # size of updates during training, preventing instability from large error jumps.
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=mem,
        processor=processor,
        nb_steps_warmup=NB_STEPS * 0.05,
        gamma=0.99,  # discount factor
        target_model_update=NB_STEPS * 0.01,
        train_interval=WINDOW_LENGTH,
        delta_clip=1,
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    # processed_obs = processor.process_observation(env.reset()[0])
    # print(
    #     "Sample observation after processing:",
    #     processed_obs.shape,
    #     processed_obs.dtype,
    # )

    return dqn


def show_model_performance(env, model, weights, epsilon):
    model.load_weights(weights)
    policy = EpsGreedyQPolicy(epsilon)
    mem = SequentialMemory(limit=NB_STEPS, window_length=WINDOW_LENGTH)
    processor = ImageProcessor()

    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        policy=policy,
        memory=mem,
        processor=processor,
    )

    dqn.compile(Adam(learning_rate=0.00025), metrics=["mae"])

    dqn.test(env, nb_episodes=1, visualize=True)

    return None


def main():
    global nb_actions

    env = GymWrapper(
        gym.make(
            ENV_NAME,
            render_mode=RENDER_MODES[2],
            repeat_action_probability=0.0,
            frameskip=WINDOW_LENGTH,
        )
    )
    # *Frame skipping is used to speed up the training process by skipping a fixed
    # number of frames (game steps) between each decision made by the agent.

    nb_actions = env.action_space.n

    np.random.seed(42)
    env.reset()

    model: Sequential = create_model()

    dqn: DQNAgent = set_agent_policy(model)

    checkpoint_callback = ModelIntervalCheckpoint(
        CHECKPOINT_FILENAME, interval=NB_STEPS * 0.1
    )

    # LOAD WEIGHTS
    # If you want to load weights from previous training session:
    # *Be sure to adjust nb_steps and policy's max epsilon accordingly.
    # *Epsilon is adjusted because you want it to be more focused on going with
    # what it knows, rather than taking random actions to try something new.
    # model.load_weights("dqn_BreakoutDeterministic-v4_weights_1200000.h5f")
    # model.load_weights(WEIGHTS_FILENAME)

    print("Logging state batch before fit() is called...")
    sample_state = env.reset()[0]  # Adjust this if needed
    sample_state = ImageProcessor().process_observation(sample_state)
    print(
        "Sample state before training:", type(sample_state), sample_state.shape
    )

    # Fit model. Choose nb_steps for training so it cuts-off at plateuing
    # and avoids overfitting.
    dqn.fit(
        env,
        nb_steps=NB_STEPS * 0.0001,
        callbacks=[checkpoint_callback],
        log_interval=500,
        visualize=False,
    )

    show_model_performance(
        env, model, "dqn_BreakoutDeterministic-v4_weights_1200000.h5f", 0.1
    )

    env.close()

    return None


# ***Training for DQNs takes a lot of time, no matter how optimized the system
# might be. The Agent still has to play the game start to finish. It can't
# break the contents of one session into chunks and process them simultaneously.
# Because of this, we should be backing our model up periodically while training.
if __name__ == "__main__":
    main()
