
#!/usr/bin/env python3
import tkinter as tk
import random

# Game configuration and constants
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BLOCK_SIZE = 30
BOARD_PIXEL_WIDTH = BOARD_WIDTH * BLOCK_SIZE
BOARD_PIXEL_HEIGHT = BOARD_HEIGHT * BLOCK_SIZE
NEXT_PREVIEW_SIZE = 4 * BLOCK_SIZE
FALL_INTERVAL = 500  # Milliseconds between automatic falls

# Tetromino shapes and corresponding colors
TETROMINO_SHAPES = {
    "I": [[1, 1, 1, 1]],
    "O": [[1, 1],
          [1, 1]],
    "T": [[0, 1, 0],
          [1, 1, 1]],
    "S": [[0, 1, 1],
          [1, 1, 0]],
    "Z": [[1, 1, 0],
          [0, 1, 1]],
    "J": [[1, 0, 0],
          [1, 1, 1]],
    "L": [[0, 0, 1],
          [1, 1, 1]]
}

TETROMINO_COLORS = {
    "I": "cyan",
    "O": "yellow",
    "T": "purple",
    "S": "green",
    "Z": "red",
    "J": "blue",
    "L": "orange"
}

def rotate_shape(shape):
    """Rotate a tetromino shape clockwise."""
    return [list(row) for row in zip(*shape[::-1])]

class Tetromino:
    def __init__(self, shape_key, shape, color):
        self.shape_key = shape_key
        self.shape = shape  # 2D list representing the tetromino
        self.color = color
        # Spawn position centered at the top row
        self.x = BOARD_WIDTH // 2 - len(shape[0]) // 2
        self.y = 0

    def rotate(self):
        self.shape = rotate_shape(self.shape)

class TetrisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Tetris")
        self.setup_widgets()

    def setup_widgets(self):
        # Main frame with some padding
        self.main_frame = tk.Frame(self.root, bg="black")
        self.main_frame.pack(padx=10, pady=10)
        # Game board canvas
        self.board_canvas = tk.Canvas(
            self.main_frame, width=BOARD_PIXEL_WIDTH, height=BOARD_PIXEL_HEIGHT, bg="white"
        )
        self.board_canvas.grid(row=0, column=0, rowspan=2, padx=10, pady=10)
        # Score label
        self.score_label = tk.Label(self.main_frame, text="Score: 0", font=("Arial", 16))
        self.score_label.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        # Next tetromino preview
        self.next_label = tk.Label(self.main_frame, text="Next:", font=("Arial", 16))
        self.next_label.grid(row=1, column=1, padx=10, pady=(20,5), sticky="nw")
        self.next_canvas = tk.Canvas(
            self.main_frame, width=NEXT_PREVIEW_SIZE, height=NEXT_PREVIEW_SIZE, bg="white"
        )
        self.next_canvas.grid(row=1, column=1, padx=10, pady=(60,10), sticky="nw")

    def update_score(self, score):
        self.score_label.config(text=f"Score: {score}")

    def draw_board(self, board):
        self.board_canvas.delete("all")
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                cell = board[y][x]
                if cell:
                    self.board_canvas.create_rectangle(
                        x * BLOCK_SIZE, y * BLOCK_SIZE,
                        (x + 1) * BLOCK_SIZE, (y + 1) * BLOCK_SIZE,
                        fill=cell, outline="black"
                    )

    def draw_next_tetromino(self, tetromino):
        self.next_canvas.delete("all")
        shape = tetromino.shape
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    self.next_canvas.create_rectangle(
                        x * BLOCK_SIZE, y * BLOCK_SIZE,
                        (x + 1) * BLOCK_SIZE, (y + 1) * BLOCK_SIZE,
                        fill=tetromino.color, outline="black"
                    )

    def display_game_over(self):
        self.board_canvas.create_text(
            BOARD_PIXEL_WIDTH // 2, BOARD_PIXEL_HEIGHT // 2,
            text="GAME OVER", fill="red", font=("Helvetica", 24)
        )

class TetrisGame:
    def __init__(self, gui):
        self.gui = gui
        self.root = gui.root
        # Initialize the game board (a grid with None in empty cells)
        self.board = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.current_tetromino = self.generate_new_tetromino()
        self.next_tetromino = self.generate_new_tetromino()
        self.bind_keys()
        self.running = True
        self.update_gui()
        self.schedule_fall()

    def generate_new_tetromino(self):
        key = random.choice(list(TETROMINO_SHAPES.keys()))
        # Make a copy of the shape so that rotations are independent per piece
        shape = [row[:] for row in TETROMINO_SHAPES[key]]
        color = TETROMINO_COLORS[key]
        return Tetromino(key, shape, color)

    def bind_keys(self):
        # Bind keyboard events to control the tetromino
        self.root.bind("<Left>", self.move_left)
        self.root.bind("<Right>", self.move_right)
        self.root.bind("<Up>", self.rotate)
        self.root.bind("<Down>", self.soft_drop)

    def move_left(self, event):
        if self.is_valid_position(self.current_tetromino, dx=-1, dy=0):
            self.current_tetromino.x -= 1
            self.update_gui()

    def move_right(self, event):
        if self.is_valid_position(self.current_tetromino, dx=1, dy=0):
            self.current_tetromino.x += 1
            self.update_gui()

    def soft_drop(self, event):
        if self.is_valid_position(self.current_tetromino, dx=0, dy=1):
            self.current_tetromino.y += 1
        else:
            self.lock_tetromino()
        self.update_gui()

    def rotate(self, event):
        original_shape = [row[:] for row in self.current_tetromino.shape]
        self.current_tetromino.rotate()
        # If new rotation is invalid, revert back
        if not self.is_valid_position(self.current_tetromino, dx=0, dy=0):
            self.current_tetromino.shape = original_shape
        self.update_gui()

    def is_valid_position(self, tetromino, dx, dy):
        new_x = tetromino.x + dx
        new_y = tetromino.y + dy
        for y, row in enumerate(tetromino.shape):
            for x, cell in enumerate(row):
                if cell:
                    board_x = new_x + x
                    board_y = new_y + y
                    # Check boundaries
                    if board_x < 0 or board_x >= BOARD_WIDTH or board_y < 0 or board_y >= BOARD_HEIGHT:
                        return False
                    # Check for collision with locked pieces
                    if board_y >= 0 and self.board[board_y][board_x]:
                        return False
        return True

    def lock_tetromino(self):
        # Add the current tetromino's blocks to the board
        for y, row in enumerate(self.current_tetromino.shape):
            for x, cell in enumerate(row):
                if cell:
                    board_x = self.current_tetromino.x + x
                    board_y = self.current_tetromino.y + y
                    if 0 <= board_y < BOARD_HEIGHT and 0 <= board_x < BOARD_WIDTH:
                        self.board[board_y][board_x] = self.current_tetromino.color
        self.clear_lines()
        # Set the next tetromino as current and generate a new next tetromino
        self.current_tetromino = self.next_tetromino
        self.next_tetromino = self.generate_new_tetromino()
        # Check for game over condition
        if not self.is_valid_position(self.current_tetromino, dx=0, dy=0):
            self.running = False
            self.gui.display_game_over()

    def clear_lines(self):
        # Identify and clear complete horizontal lines
        full_lines = [i for i, row in enumerate(self.board) if all(row)]
        for i in full_lines:
            del self.board[i]
            self.board.insert(0, [None for _ in range(BOARD_WIDTH)])
        if full_lines:
            self.score += 100 * len(full_lines)
            self.gui.update_score(self.score)

    def fall(self):
        if self.running:
            if self.is_valid_position(self.current_tetromino, dx=0, dy=1):
                self.current_tetromino.y += 1
            else:
                self.lock_tetromino()
            self.update_gui()
            self.schedule_fall()

    def schedule_fall(self):
        self.root.after(FALL_INTERVAL, self.fall)

    def update_gui(self):
        # Create a temporary board which includes the locked pieces and the current tetromino's position
        temp_board = [row[:] for row in self.board]
        for y, row in enumerate(self.current_tetromino.shape):
            for x, cell in enumerate(row):
                if cell:
                    board_x = self.current_tetromino.x + x
                    board_y = self.current_tetromino.y + y
                    if 0 <= board_y < BOARD_HEIGHT and 0 <= board_x < BOARD_WIDTH:
                        temp_board[board_y][board_x] = self.current_tetromino.color
        self.gui.draw_board(temp_board)
        self.gui.draw_next_tetromino(self.next_tetromino)

if __name__ == "__main__":
    root = tk.Tk()
    gui = TetrisGUI(root)
    game = TetrisGame(gui)
    root.mainloop()

