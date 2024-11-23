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
        self.char = "o"


    def get_move(self, event_position):
        # TODO: lab3 - implement algorithm
        current_board = self.game.board
        value = self.minimax(current_board)
        raise NotImplementedError


    def get_board_after_move(self, move, board):
        board[move] = self.char
        return board


    def evaluate_board(self, board):
        winner = self.game.get_winner()
        match winner:
            case "" | "t":
                return 0
            case self.char:
                print("win")
                return 1
            case _:
                print("lose")
                return -1


    def minimax(self, current_board):
        available_moves = self.game.available_moves()
        best_value = -np.inf
        best_move = None

        for move in available_moves:
            board_copy = current_board.copy()
            board_after_move = self.get_board_after_move(move, board_copy)
            value = self.evaluate_board(board_after_move)
            #print(value)
            if value == 0:
                value, move = self.minimax(board_after_move)
            if value > best_value:
                best_value = value
                print("aaaaaaa")
                best_move = move

        return best_value, best_move




