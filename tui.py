import numpy as np
import time

class GameTUI:
    def __init__(self, game, player_x, player_o, endless=False):
        self.game = game
        self.player_x = player_x
        self.player_o = player_o
        self.ties = 0
        self.endless = endless

    def mainloop(self):
        current_player = None
        while True:
            self.draw_board(self.game.board)
            if self.game.get_winner() in ["x", "o"]:
                if current_player:
                    current_player.score += 1
                print(f" {self.game.get_winner()} wins!!!\n")
                break
            elif self.game.get_winner() == "t":
                self.ties += 1
                print(" Its a draw!\n")
                break
            current_player = self.player_x if self.game.player_x_turn else self.player_o
            print(f" {current_player.char}'s turn!")
            if current_player.is_human:
                row, column = self.get_human_move()
            else:
                row, column = self.get_ai_move()
            self.place_char(row, column, current_player.char)
            self.game.player_x_turn = not self.game.player_x_turn
            time.sleep(1)
        self.post_match()

    def draw_row(self, row):
        row_str = "   |"
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
        while True:
            try:
                row = int(input("Enter row (1-3): "))
                column = int(input("Enter column (1-3): "))
                if row < 1 or row > 3 or column < 1 or column > 3 or self.game.board[row-1, column-1] != "":
                    raise ValueError
                break
            except ValueError:
                print("\nInvalid input. Choose unoccupied space entering numbers between 1 and 3!\n")
        return row-1, column -1

    def get_ai_move(self):
        if self.game.player_x_turn:
            return self.player_x.get_move()
        else:
            return self.player_o.get_move()

    def place_char(self, row, column, char):
        self.game.board[row, column] = char

    def post_match(self):
        print("===SCORE===")
        print(f"x wins: {self.player_x.score} / o wins: {self.player_o.score} / draws: {self.ties}\n")
        if not self.endless:
            play_again = input("Play again? (Y/N): ")
            if play_again.upper() == "Y":
                self.game.play_again()
                self.mainloop()
            else:
                print("Thanks for playing! Bye!\n")
        else:
            self.game.play_again()
            self.mainloop()




