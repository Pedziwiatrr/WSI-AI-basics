import argparse
import gymnasium as gym
from cliff_walking import q_learn
from visualizer import reward_plot, postprocess, plot_q_values_map, plot_states_actions_distribution, analyze_rewards


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

    average_reward, best_reward, average_reward_last_100 = analyze_rewards(rewards_list)
    print(">=====================< FINISHED >=====================<")
    print("> Params:")
    print(f"    Episodes: {args.episodes}")
    print(f"    Max steps per episode: {args.max_steps}")
    print(f"    Exploration rate: {args.epsilon}")
    print(f"    Learning rate: {args.beta}")
    print(f"    Discount factor: {args.gamma}")
    print(">------------------------------------------------------<")
    print("> Results:")
    print(f"    Average reward: {average_reward:.3f}")
    print(f"    Average reward in last 100 episodes: {average_reward_last_100:.3f}")
    print(f"    Best reward: {best_reward}  (Best possible: -13)")
    print(">======================================================<")

    reward_plot(list(range(1, args.episodes + 1)), rewards_list, 'plots/reward_plot.png')
    plot_q_values_map(Q, environment, "plots/q_values_map.png")
    plot_states_actions_distribution(states, actions, 'plots/states_actions_distribution.png')

if __name__ == '__main__':
    main()