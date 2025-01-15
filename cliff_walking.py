import numpy as np



def q_learn(environment, episode_count=10, max_steps=1000, beta=0.8, gamma=0.1, epsilon=0.2):
    episode = 0
    # Q will store possible rewards for every action in every state. We don't know the values yet so right now its just np.zeros
    Q = np.zeros((environment.observation_space.n, environment.action_space.n))                     # Q ← init

    # values history needed for visualization
    rewards_list = []
    steps_list = []
    states = []
    actions = []

    while episode < episode_count:
        state, info = environment.reset()
        steps = 0
        sum_reward = 0
        while steps < max_steps:
            action = choose_action(Q, state, epsilon, environment.action_space.n)                   # at ← choose action (xt,Qt)
            next_state, reward, terminated, truncated, info = environment.step(action)              # rt, xt+1 ← do at

            # delta will store the error between our current state and the next state
            # predicted future reward is discounted by gamma because
            # we don't want it to impact our decisions too much as it might be incorrect.
            delta = reward + gamma * np.max(Q[next_state, :]) - Q[state, action]  # ∆ ← rt + γ max Qt(xt+1,a) − Qt(xt,at)
            Q[state, action] += beta * delta                                                        # Qt+1 ← Qt + β∆

            state = next_state
            sum_reward += reward
            steps += 1                                                                              # t ← t + 1

            states.append(state)
            actions.append(action)

            if terminated or truncated:
                break

        episode += 1

        rewards_list.append(sum_reward)
        steps_list.append(steps)
        print(f'>> Finished episode: {episode}, steps: {steps}, Reward sum: {sum_reward} <<')

    return Q, rewards_list, steps_list, states, actions


def choose_action(Q, state, epsilon, possible_actions):
    if np.random.rand() < epsilon:
        # exploration: trying out new actions that are not yet explored
        action = np.random.choice(possible_actions)
    else:
        # exploitation: choosing the best explored action
        action = np.argmax(Q[state, :])
    return action

