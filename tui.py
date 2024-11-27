import numpy as np


class GameTUI:
    def __init__(self, game, player_x, player_o):
        self.game = game
        self.player_x = player_x
        self.player_o = player_o

    def mainloop(self):
        while True:
            self.draw_board(self.game.board)
            if self.game.get_winner() != "":
                print(f"THE WINNER IS: {self.game.get_winner()}!!!")
                break
            else:
                print(self.game.get_winner())
            current_player = self.player_x if self.game.player_x_turn else self.player_o
            print(f"It is {current_player.char}'s turn!")
            if current_player.is_human:
                column, row = self.get_human_move()
            else:
                column, row = self.get_ai_move()
            self.place_char(column, row, current_player.char)
            self.game.player_x_turn = not self.game.player_x_turn

    def draw_row(self, row):
        row_str = "|"
        for char in row:
            if char == "":
                row_str += " "
            else:
                row_str += char
            row_str += "|"
        print(row_str)

    def draw_board(self, board):
        print()
        for row in board:
            self.draw_row(row)
        print()

    def get_human_move(self):
        print("Your turn!")
        while True:
            try:
                column = int(input("Enter column (1-3): "))
                row = int(input("Enter row (1-3): "))
                break
            except ValueError:
                print("Invalid input. Enter numbers between 1 and 3")
        return column -1, row-1

    def get_ai_move(self):
        if self.game.player_x_turn:
            return self.player_x.get_move()
        else:
            return self.player_o.get_move()

    def place_char(self, column, row, char):
        self.game.board[row, column] = char




