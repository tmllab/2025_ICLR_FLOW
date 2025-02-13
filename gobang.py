# Gobang Game Implementation in Python

import tkinter as tk
from tkinter import messagebox
import random

class GobangGame:
    def __init__(self):
        self.board_size = 15
        self.board = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_turn = 'human'  # or 'AI'
        self.winner = None
        self.player_color = None

    def initialize_game(self):
        self.board = [[None for _ in range(self.board_size)] for _ in range(self.board_size)]
        self.current_turn = 'human'  # Reset to human's turn
        self.winner = None

    def check_game_status(self):
        if self.check_winner():
            return 'winner', self.winner
        elif all(all(cell is not None for cell in row) for row in self.board):
            return 'draw', None
        return 'ongoing', None

    def check_winner(self):
        # Check horizontal, vertical, and diagonal for a winner
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row][col] is not None:
                    if self.check_direction(row, col, 1, 0) or \
                       self.check_direction(row, col, 0, 1) or \
                       self.check_direction(row, col, 1, 1) or \
                       self.check_direction(row, col, 1, -1):
                        self.winner = self.board[row][col]
                        return True
        return False

    def check_direction(self, row, col, delta_row, delta_col):
        count = 0
        color = self.board[row][col]
        for _ in range(5):
            if 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row][col] == color:
                count += 1
                row += delta_row
                col += delta_col
            else:
                break
        return count == 5

    def make_move(self, row, col):
        if self.board[row][col] is None:
            self.board[row][col] = 'black' if self.current_turn == 'human' else 'white'
            self.switch_turn()

    def switch_turn(self):
        self.current_turn = 'AI' if self.current_turn == 'human' else 'human'

class AIPlayer:
    def __init__(self, game):
        self.game = game

    def make_move(self):
        # Simple AI that makes a random valid move
        empty_cells = [(r, c) for r in range(self.game.board_size) for c in range(self.game.board_size) if self.game.board[r][c] is None]
        if empty_cells:
            row, col = random.choice(empty_cells)
            self.game.make_move(row, col)

class GobangGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Gobang Game")
        self.canvas = tk.Canvas(master, width=600, height=600, bg="#f0f0f0")
        self.canvas.pack()
        self.game = GobangGame()
        self.ai_player = AIPlayer(self.game)
        self.cell_size = 40
        self.draw_board()
        self.canvas.bind("<Button-1>", self.handle_click)
        self.turn_label = tk.Label(master, text="Current Turn: Human (Black)", font=("Arial", 24))
        self.turn_label.pack()

    def draw_board(self):
        for i in range(self.game.board_size):
            self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, self.cell_size * self.game.board_size, fill="#cccccc")
            self.canvas.create_line(0, i * self.cell_size, self.cell_size * self.game.board_size, i * self.cell_size, fill="#cccccc")

    def handle_click(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if self.game.current_turn == 'human':
            self.game.make_move(row, col)
            self.update_board()
            self.check_game_status()
            if self.game.current_turn == 'AI':
                self.ai_player.make_move()
                self.update_board()
                self.check_game_status()

    def update_board(self):
        self.canvas.delete("all")
        self.draw_board()
        for r in range(self.game.board_size):
            for c in range(self.game.board_size):
                if self.game.board[r][c] == 'black':
                    self.canvas.create_oval(c * self.cell_size + 5, r * self.cell_size + 5,
                                            c * self.cell_size + self.cell_size - 5, r * self.cell_size + self.cell_size - 5,
                                            fill="black")
                elif self.game.board[r][c] == 'white':
                    self.canvas.create_oval(c * self.cell_size + 5, r * self.cell_size + 5,
                                            c * self.cell_size + self.cell_size - 5, r * self.cell_size + self.cell_size - 5,
                                            fill="white")
        self.turn_label.config(text=f"Current Turn: {self.game.current_turn.capitalize()}")

    def check_game_status(self):
        status, winner = self.game.check_game_status()
        if status == 'winner':
            messagebox.showinfo("Game Over", f"Congratulations! {winner.capitalize()} wins!")
            self.game.initialize_game()
            self.update_board()
        elif status == 'draw':
            messagebox.showinfo("Game Over", "The game is a draw! The board is full.")
            self.game.initialize_game()
            self.update_board()

if __name__ == "__main__":
    root = tk.Tk()
    gobang_gui = GobangGUI(root)
    root.mainloop()