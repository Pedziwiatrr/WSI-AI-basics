import gymnasium as gym
import numpy as np

def q_learn(environment, episode_count=10, max_steps=1000, beta=0.8, gamma=0.1):
    episode = 0
    while episode < episode_count:
        sum_reward = 0
        state, info = environment.reset()
        steps = 0                                                                                   # t ← 0
        Q = np.zeros((environment.observation_space.n, environment.action_space.n))                 # Q ← init
        while steps < max_steps:
            action = choose_action()                                                                # at ← choose action (xt,Qt)
            next_state, reward, is_finished, is_finished, info =  environment.step(action)          # rt, xt+1 ← do at
            delta = reward + gamma * np.max(Q[next_state, :]) - Q[state, action]                    # ∆ ← rt + γ max Qt(xt+1,a) − Qt(xt,at)
            Q[state, action] = beta * delta                                                         # Qt+1 ← Qt + β∆
            sum_reward += reward
            steps += 1                                                                              # t ← t + 1
        episode += 1


def choose_action():
    return 0

