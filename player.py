from abc import ABC, abstractmethod

import numpy as np


def build_player(player_config, game):
    assert player_config["type"] in ["human", "random", "minimax"]

    if player_config["type"] == "human":
        return HumanPlayer(game)

    if player_config["type"] == "random":
        return RandomComputerPlayer(game)

    if player_config["type"] == "minimax":
        return MinimaxComputerPlayer(game, player_config)


class Player(ABC):
    def __init__(self, game):
        self.game = game
        self.score = 0

    @abstractmethod
    def get_move(self, event_position):
        pass


class HumanPlayer(Player):
    def get_move(self, event_position):
        return event_position


class RandomComputerPlayer(Player):
    def get_move(self, event_position):
        available_moves = self.game.available_moves()
        move_id = np.random.choice(len(available_moves))
        return available_moves[move_id]


class MinimaxComputerPlayer(Player):
    def __init__(self, game, config):
        super().__init__(game)
        # TODO: lab3 - load pruning depth from config
        self.depth = config["depth"]

    def get_move(self, event_position):
        # TODO: lab3 - implement algorithm
        value = self.minimax_evaluate(event_position)
        raise NotImplementedError

    def minimax_evaluate(self, event_position):
        available_moves = self.game.available_moves()
        best_value = -np.inf
        best_move = None

        for move in available_moves:
            print(move)

        return best_value

