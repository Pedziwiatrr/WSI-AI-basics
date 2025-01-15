from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


sns.set_theme()



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
    directions = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    qtable_directions = np.empty(qtable_best_action.flatten().shape, dtype=str)
    eps = np.finfo(float).eps
    for idx, val in enumerate(qtable_best_action.flatten()):
        if qtable_val_max.flatten()[idx] > eps:
            qtable_directions[idx] = directions[val]
    return qtable_val_max, qtable_directions


def plot_q_values_map(qtable, env, savefig_folder="q_values_map.png"):
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

    img_title = f"cliffwalking_q_values.png"
    fig.savefig(savefig_folder, bbox_inches="tight")
    plt.show()



