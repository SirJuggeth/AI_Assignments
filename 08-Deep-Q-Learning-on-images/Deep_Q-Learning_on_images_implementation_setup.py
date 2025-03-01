from collections import deque

import ale_py  # is being accessed even though pylance says it isn't.
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.utils import play
from PIL import Image
from rl.core import Processor
from rl.memory import SequentialMemory

WINDOW_LENGTH = 3  # More complex games will use a larger window length
IMG_SHAPE = (84, 84)

RENDER_MODES = [
    None,
    "human",
    "rgb_array",
    "ansi",
    "rgb_array_list",
    "human_pixels",
]

ENV_NAME = "ALE/Breakout-v5"

env = gym.make(ENV_NAME, render_mode=RENDER_MODES[2])


class BreakOutProcessor(Processor):
    def process_observation(self, observation: np.ndarray):
        # Convert observation from numpy.ndarray from the sequntial_frame_buffer
        # to  PIL Image so it will work with the Python Imaging Library
        img = Image.fromarray(observation)

        # convert Breakout obs shape from (210, 160, 3)
        # 210 height x 160 width with 3 colour channels RGB
        # to (84,84) to reduce resolution for quicker processing.
        img = img.resize(IMG_SHAPE)

        # Convert from 3 channel RGB colour to single channel grayscale
        # (L stands for luminescence)
        img = img.convert("L")

        # convert back to np.ndarray
        return np.array(img)


def main():
    np.random.seed(42)
    env.reset()

    """
    # Quick test to show env activity by grabbing sequential frames manually
    sequential_frame_buffer = []
    temp_seqential_frames = deque(maxlen=WINDOW_LENGTH)

    for i in range(44):
        if i == 1:
            action = 1  # Initiate ball
        else:
            action = env.action_space.sample()
            print("Action: ", action)

        obs, rew, done, trunc, info = env.step(action)

        if len(temp_seqential_frames) == WINDOW_LENGTH:
            # Once window length has been met, start appending frames to buffer.
            sequential_frame_buffer.append(list(temp_seqential_frames))

        temp_seqential_frames.append(obs)
    """

    """
    # Another way to show env activity (but not sure how it works
    # or what the current method is for achieving this)
    plt.imshow(sequential_frame_buffer[0][0])

    fig, axis = plt.subplots(444, WINDOW_LENGTH, figsize=(12, 12))

    for global_index, timestep in enumerate(sequential_frame_buffer[:444]):
        for frame_index, frame in enumerate(timestep):
            axis[global_index][frame_index].imshow(frame)
    fig.subplots_adjust(wspace=0, hspace=0.1)
    """

    # Different way to do the same thing, apparently
    # mem = SequentialMemory(limit=1000, window_length=WINDOW_LENGTH)

    """
    # Quick test to show BreakoutProcessor working
    sample_imgs = []
    breakout_proc = BreakOutProcessor()
    env.reset()

    for i in range(200):
        print("Breakout test, step: ", i)
        action = env.action_space.sample()
        obs, r, d, trunc, info = env.step(action)
        # Reduce obs (frame) resolution and convert from RGB to grayscale
        processed_obs = breakout_proc.process_observation(obs)
        sample_imgs.append(processed_obs)

    print("\nsample_imgs list size: ", len(sample_imgs))
    print("Processed image shape: ", sample_imgs[-1].shape)
    print("Processed image array contents: ", sample_imgs[-1])
    # plt.imshow(sample_imgs[-1])
    plt.imshow(sample_imgs[-1], cmap="gray")
    plt.show()
    """

    env.close()

    return None


if __name__ == "__main__":
    main()
