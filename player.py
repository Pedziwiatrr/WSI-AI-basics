from abc import ABC, abstractmethod
import copy
import numpy as np


def build_player(player_config, game, char=None):
    assert player_config["type"] in ["human", "random", "minimax"]

    if player_config["type"] == "human":
        return HumanPlayer(game)

    if player_config["type"] == "random":
        return RandomComputerPlayer(game)

    if player_config["type"] == "minimax":
        return MinimaxComputerPlayer(game, char, player_config)


class Player(ABC):
    def __init__(self, game):
        self.game = game
        self.score = 0

    @abstractmethod
    def get_move(self, event_position):
        pass


class HumanPlayer(Player):
    def get_move(self, event_position=None):
        return event_position


class RandomComputerPlayer(Player):
    def get_move(self, event_position=None):
        available_moves = self.game.available_moves()
        move_id = np.random.choice(len(available_moves))
        #print(available_moves[move_id])
        return available_moves[move_id]


class MinimaxComputerPlayer(Player):
    def __init__(self, game, char, config):
        super().__init__(game)
        # TODO: lab3 - load pruning depth from config
        self.depth = config["depth"]
        self.char = char


    def get_move(self, event_position=None):
        # TODO: lab3 - implement algorithm
        current_board = self.game.board
        value, move = self.minimax(current_board, True)
        if move is None:
            print("Move is None, choosing random move...")
            available_moves = self.game.available_moves()
            move_id = np.random.choice(len(available_moves))
            move = available_moves[move_id]
        return move


    def minimax_simulate_move(self, move, board, is_maximizing):
        board_copy = copy.deepcopy(board)
        if self.char == "o":
            if is_maximizing:
                board_copy[move[0], move[1]] = "o"
            elif not is_maximizing:
                board_copy[move[0], move[1]] = "x"
        elif self.char == "x":
            if is_maximizing:
                board_copy[move[0], move[1]] = "x"
            elif not is_maximizing:
                board_copy[move[0], move[1]] = "o"
        return board_copy


    def evaluate_board(self, board):
        winner = self.game.get_winner(board)
        if winner == "" or winner == "t":
            return 0
        elif winner == self.char: return 1
        else: return -1


    def minimax(self, board, is_maximizing, depth=0, alpha=-np.inf, beta=np.inf):
        if is_maximizing: best_value = -np.inf
        else: best_value = np.inf
        best_move = None
        available_moves = self.game.available_moves(board)
        if depth == self.depth or self.game.get_winner(board) != "":
            return self.evaluate_board(board, is_maximizing), None

        for move in available_moves:
            board_after_move = self.minimax_simulate_move(move, board, is_maximizing)
            value, next_move = self.minimax(board_after_move, not is_maximizing, depth+1)
            if (is_maximizing and value > best_value) or (not is_maximizing and value < best_value):
                best_value = value
                best_move = move
                if is_maximizing:
                    alpha = max(alpha, best_value)
                else:
                    beta = min(beta, best_value)
                if beta <= alpha:
                    break

        print(f"Depth: {depth}, Best Value: {best_value}, Best Move: {best_move}, is_maximizing: {is_maximizing}")
        return best_value, best_move




