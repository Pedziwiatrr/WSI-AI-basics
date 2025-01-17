import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set_theme()


def reward_plot(episodes, rewards, savefig_file='plots/reward_plot.png'):
    plt.figure(figsize=(15, 5))
    plt.plot(episodes, rewards, linestyle='-', color='b', label='Reward')

    model = LinearRegression()
    model.fit(np.array(episodes).reshape(-1, 1), rewards)
    trend = model.predict(np.array(episodes).reshape(-1, 1))
    plt.plot(episodes, trend, linestyle='-', color='r', label='Trend Line')

    plt.title('Reward over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.ylim(-1000)
    plt.legend()
    plt.grid(True)
    plt.savefig(savefig_file)

def postprocess(episodes, rewards, steps):
    res = pd.DataFrame(
        data={
            "Episodes": np.tile(episodes, len(rewards) // len(episodes)),
            "Rewards": rewards,
            "Steps": steps,
        }
    )
    res["cum_rewards"] = res["Rewards"].cumsum()

    st = pd.DataFrame(data={"Episodes": episodes, "Steps": np.mean(steps)})
    return res, st

def qtable_directions_map(qtable):
    qtable_val_max = qtable.max(axis=1)
    qtable_best_action = np.argmax(qtable, axis=1)
    #directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    directions = {0: "↑", 1: "→", 2: "↓", 3: "←"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] != 0:
            qtable_directions[idx] = directions[val]
    return qtable_val_max, qtable_directions


def plot_states_actions_distribution(states, actions, savefig_file='plots/states_actions_distribution.png'):
    labels = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    sns.histplot(data=states, ax=ax[0], kde=True)
    ax[0].set_title("States")
    sns.histplot(data=actions, ax=ax[1])
    ax[1].set_xticks(list(labels.values()), labels=labels.keys())
    ax[1].set_title("Actions")
    fig.tight_layout()
    fig.savefig(savefig_file, bbox_inches="tight")


def plot_q_values_map(qtable, env, savefig_file="q_values_map.png"):
    qtable_val_max, qtable_directions = qtable_directions_map(qtable)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    ax[0].imshow(env.render())
    ax[0].axis("off")
    ax[0].set_title("Last frame")

    sns.heatmap(
        qtable_val_max.reshape(4, 12),
        annot=qtable_directions.reshape(4, 12),
        fmt="",
        ax=ax[1],
        cmap=sns.color_palette("Blues", as_cmap=True),
        linewidths=0.7,
        linecolor="black",
        xticklabels=[],
        yticklabels=[],
        annot_kws={"fontsize": "xx-large"},
    ).set(title="Learned Q-values\nArrows represent best action")

    for _, spine in ax[1].spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.7)
        spine.set_color("black")
    fig.savefig(savefig_file, bbox_inches="tight")



