import argparse
import json
import pathlib

import numpy as np

from gui import GameGUI
from game import TicTacToe
from player import build_player
from tui import GameTUI


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=pathlib.Path, required=True, help="Path to game config")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    with open(args.config, "r") as f:
        config = json.load(f)

    game = TicTacToe()
    player_x = build_player(config["x"], game, "x")
    player_o = build_player(config["o"], game, "o")

    if config["gui"]:
        gui = GameGUI(game, player_x, player_o)
        gui.mainloop()
    else:
        tui = GameTUI(game, player_x, player_o, config["endless"])
        tui.mainloop()


