import time
from turtle import window_width

import gymnasium as gym
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from tensorflow.keras.layers import Activation, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

ENV_NAME = "CartPole-v1"

env = gym.make(ENV_NAME, render_mode=None)
env.reset()


nb_actions = env.action_space.n  # number of actions
nb_obs = env.observation_space.shape  # number of obs
print("\n\n", nb_obs, "\n\n")


def build_training_model(summ: bool = False):
    model = Sequential()

    model.add(Flatten(input_shape=(1,) + nb_obs))

    model.add(Dense(16))
    model.add(Activation("relu"))

    model.add(Dense(32))
    model.add(Activation("relu"))

    model.add(Dense(nb_actions))
    model.add(Activation("linear"))

    if summ:
        print(model.summary())

    return model


def __env_test():
    ### Quick Env test ###
    env = gym.make(ENV_NAME, render_mode="human")
    env.reset()

    for steps in range(100):
        action = env.action_space.sample()
        env.step(action)

        env.render()
        time.sleep(0.05)

    env.close()

    return None


def main():
    model = build_training_model()  # (summ=True)

    mem = SequentialMemory(limit=20000, window_length=1)

    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr="eps",
        value_max=1.0,
        value_min=0.1,
        value_test=0.05,
        nb_steps=20000,
    )
    dqn = DQNAgent(
        model=model,
        nb_actions=nb_actions,
        memory=mem,
        nb_steps_warmup=10,
        target_model_update=100,
        policy=policy,
    )

    dqn.compile(Adam(learning_rate=1e-3), metrics=["mae"])
    dqn.fit(env, nb_steps=20000, visualize=False, verbose=2)

    dqn.save_weights(f"model_weights_cartpole.h5f", overwrite=True)

    dqn.test(env, nb_episodes=5, visualize=True)

    env.close()


if __name__ == "__main__":
    # __env_test()
    main()
