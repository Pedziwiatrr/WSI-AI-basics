import gymnasium as gym
import numpy as np


def q_learn(environment, episode_count=10, max_steps=1000, beta=0.8, gamma=0.1):
    for episode in range(episode_count):
        environment.reset()
        steps = 0                                                                       # t ← 0
        Q = np.zeros((environment.observation_space.n, environment.action_space.n))     # Q ← init
        while steps < max_steps:
            action = get_action()                                                       # at ← get action(xt,Qt)
            do_action(action)                                                           # rt, xt+1 ← do at
            steps += 1                                                                  # t ← t + 1


def get_action():
    pass

def do_action(action):
    pass