import pygame
import random
import sys

# Initialize pygame
pygame.init()

# Screen dimensions and grid parameters
CELL_SIZE = 30
COLS = 10
ROWS = 20
SIDE_PANEL = 150  # space for score and next tetromino
WIDTH = COLS * CELL_SIZE + SIDE_PANEL
HEIGHT = ROWS * CELL_SIZE
FPS = 60

# Colors (R, G, B)
BLACK   = (0, 0, 0)
WHITE   = (255, 255, 255)
GRAY    = (128, 128, 128)
RED     = (255, 0, 0)
GREEN   = (0, 255, 0)
BLUE    = (0, 0, 255)
CYAN    = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW  = (255, 255, 0)
ORANGE  = (255, 165, 0)

# Define tetromino shapes and their rotations
TETROMINOES = {
    'I': [
        [[1, 1, 1, 1]],
        [[1],
         [1],
         [1],
         [1]]
    ],
    'J': [
        [[1, 0, 0],
         [1, 1, 1]],
        [[1, 1],
         [1, 0],
         [1, 0]],
        [[1, 1, 1],
         [0, 0, 1]],
        [[0, 1],
         [0, 1],
         [1, 1]]
    ],
    'L': [
        [[0, 0, 1],
         [1, 1, 1]],
        [[1, 0],
         [1, 0],
         [1, 1]],
        [[1, 1, 1],
         [1, 0, 0]],
        [[1, 1],
         [0, 1],
         [0, 1]]
    ],
    'O': [
        [[1, 1],
         [1, 1]]
    ],
    'S': [
        [[0, 1, 1],
         [1, 1, 0]],
        [[1, 0],
         [1, 1],
         [0, 1]]
    ],
    'T': [
        [[0, 1, 0],
         [1, 1, 1]],
        [[1, 0],
         [1, 1],
         [1, 0]],
        [[1, 1, 1],
         [0, 1, 0]],
        [[0, 1],
         [1, 1],
         [0, 1]]
    ],
    'Z': [
        [[1, 1, 0],
         [0, 1, 1]],
        [[0, 1],
         [1, 1],
         [1, 0]]
    ]
}

# Colors for each tetromino type
TETROMINO_COLORS = {
    'I': CYAN,
    'J': BLUE,
    'L': ORANGE,
    'O': YELLOW,
    'S': GREEN,
    'T': MAGENTA,
    'Z': RED
}

# Game clock
clock = pygame.time.Clock()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tetris")

# Font for UI
font = pygame.font.SysFont("comicsans", 24)

def create_grid(locked_positions={}):
    grid = [[BLACK for _ in range(COLS)] for _ in range(ROWS)]
    for r in range(ROWS):
        for c in range(COLS):
            if (c, r) in locked_positions:
                grid[r][c] = locked_positions[(c, r)]
    return grid

class Tetromino:
    def __init__(self, shape):
        self.shape = shape
        self.rotations = TETROMINOES[shape]
        self.rotation = 0
        self.matrix = self.rotations[self.rotation]
        # Spawn at the top middle
        self.x = COLS // 2 - len(self.matrix[0]) // 2
        self.y = 0
        self.color = TETROMINO_COLORS[shape]

    def rotate(self, grid):
        new_rotation = (self.rotation + 1) % len(self.rotations)
        new_matrix = self.rotations[new_rotation]
        if not self.check_collision(self.x, self.y, new_matrix, grid):
            self.rotation = new_rotation
            self.matrix = new_matrix

    def check_collision(self, x_offset, y_offset, matrix, grid):
        for y, row in enumerate(matrix):
            for x, cell in enumerate(row):
                if cell:
                    new_x = x + x_offset
                    new_y = y + y_offset
                    if new_x < 0 or new_x >= COLS or new_y >= ROWS:
                        return True
                    if new_y >= 0 and grid[new_y][new_x] != BLACK:
                        return True
        return False

    def move(self, dx, grid):
        if not self.check_collision(self.x + dx, self.y, self.matrix, grid):
            self.x += dx

    def fall(self, grid):
        if not self.check_collision(self.x, self.y + 1, self.matrix, grid):
            self.y += 1
            return True
        return False

    def lock(self, locked_positions):
        for y, row in enumerate(self.matrix):
            for x, cell in enumerate(row):
                if cell:
                    pos = (self.x + x, self.y + y)
                    locked_positions[pos] = self.color

def clear_rows(grid, locked_positions):
    cleared = 0
    for i in range(ROWS-1, -1, -1):
        if BLACK not in grid[i]:
            cleared += 1
            # Remove positions in row i from locked_positions
            for j in range(COLS):
                try:
                    del locked_positions[(j, i)]
                except:
                    continue
            # Shift every row above i down
            for key in sorted(list(locked_positions), key=lambda x: x[1])[::-1]:
                x, y = key
                if y < i:
                    locked_positions[(x, y + 1)] = locked_positions.pop(key)
    return cleared

def draw_grid(surface, grid):
    for i in range(ROWS):
        for j in range(COLS):
            pygame.draw.rect(surface, grid[i][j], (j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)
            # Grid border lines
            pygame.draw.rect(surface, GRAY, (j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

def draw_side_panel(surface, score, next_piece):
    # Draw side panel background
    panel_rect = (COLS*CELL_SIZE, 0, SIDE_PANEL, HEIGHT)
    pygame.draw.rect(surface, BLACK, panel_rect)
    # Score text
    score_text = font.render("Score:", True, WHITE)
    surface.blit(score_text, (COLS*CELL_SIZE + 20, 20))
    score_val = font.render(str(score), True, WHITE)
    surface.blit(score_val, (COLS*CELL_SIZE + 20, 50))
    # Next piece text
    next_text = font.render("Next:", True, WHITE)
    surface.blit(next_text, (COLS*CELL_SIZE + 20, 100))
    # Draw next tetromino preview
    matrix = next_piece.rotations[0]  # default rotation for preview
    for y, row in enumerate(matrix):
        for x, cell in enumerate(row):
            if cell:
                pygame.draw.rect(surface, next_piece.color, (COLS*CELL_SIZE + 20 + x*CELL_SIZE,
                                                              130 + y*CELL_SIZE, CELL_SIZE, CELL_SIZE), 0)
                pygame.draw.rect(surface, GRAY, (COLS*CELL_SIZE + 20 + x*CELL_SIZE,
                                                 130 + y*CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

def draw_window(surface, grid, score, next_piece):
    surface.fill(BLACK)
    draw_grid(surface, grid)
    draw_side_panel(surface, score, next_piece)
    pygame.display.update()

def main():
    locked_positions = {}
    grid = create_grid(locked_positions)

    current_piece = Tetromino(random.choice(list(TETROMINOES.keys())))
    next_piece = Tetromino(random.choice(list(TETROMINOES.keys())))
    fall_time = 0
    fall_speed = 0.5  # seconds per grid fall
    score = 0

    running = True
    while running:
        dt = clock.tick(FPS) / 1000  # seconds elapsed since last frame
        fall_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.move(-1, grid)
                elif event.key == pygame.K_RIGHT:
                    current_piece.move(1, grid)
                elif event.key == pygame.K_UP:
                    current_piece.rotate(grid)
                elif event.key == pygame.K_DOWN:
                    # Accelerate fall when pressing down
                    if current_piece.fall(grid):
                        fall_time = 0

        if fall_time >= fall_speed:
            fall_time = 0
            if not current_piece.fall(grid):
                # Lock piece if it can’t move down
                current_piece.lock(locked_positions)
                grid = create_grid(locked_positions)
                # Clear rows and update score (e.g., 100 pts per row)
                cleared = clear_rows(grid, locked_positions)
                if cleared:
                    score += cleared * 100
                # Switch to next piece and generate a new preview
                current_piece = next_piece
                next_piece = Tetromino(random.choice(list(TETROMINOES.keys())))
                # Check if game over
                if current_piece.check_collision(current_piece.x, current_piece.y, current_piece.matrix, grid):
                    print("Game Over! Score:", score)
                    running = False

        grid = create_grid(locked_positions)
        # Draw current falling tetromino onto grid
        for y, row in enumerate(current_piece.matrix):
            for x, cell in enumerate(row):
                if cell:
                    pos_x = current_piece.x + x
                    pos_y = current_piece.y + y
                    if pos_y >= 0:
                        grid[pos_y][pos_x] = current_piece.color

        draw_window(screen, grid, score, next_piece)

    pygame.time.delay(2000)

if __name__ == '__main__':
    main()
