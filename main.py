import argparse
import gymnasium as gym
from cliff_walking import q_learn
from visualizer import reward_plot, postprocess, plot_q_values_map, plot_states_actions_distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--epsilon', type=float, default=0.0001)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.9)
    args = parser.parse_args()

    environment = gym.make("CliffWalking-v0", render_mode="rgb_array")
    Q, rewards_list, steps_list, states, actions = q_learn(environment, args.episodes, args.max_steps, args.beta, args.gamma, args.epsilon)

    episodes = list(range(1, args.episodes + 1))
    results, summary = postprocess(episodes, rewards_list, steps_list)

    reward_plot(episodes, rewards_list, 'plots/reward_plot.png')
    plot_q_values_map(Q, environment, "plots/q_values_map.png")
    plot_states_actions_distribution(states, actions, 'plots/states_actions_distribution.png')

if __name__ == '__main__':
    main()