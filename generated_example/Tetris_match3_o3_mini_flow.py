
import pygame
import random
import sys

# Configuration Constants
CELL_SIZE = 30
GRID_COLS, GRID_ROWS = 10, 20
SIDE_PANEL_WIDTH = 150
WINDOW_WIDTH = CELL_SIZE * GRID_COLS + SIDE_PANEL_WIDTH
WINDOW_HEIGHT = CELL_SIZE * GRID_ROWS
FPS = 60

# Scoring Constants
BASE_LINE_CLEAR_POINTS = 100
BASE_MATCH3_POINTS = 50

# Colors
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
WHITE = (255, 255, 255)
RED = (220, 20, 60)
GREEN = (34, 139, 34)
BLUE = (30, 144, 255)
YELLOW = (238, 232, 170)
CYAN = (0, 206, 209)
MAGENTA = (218, 112, 214)
ORANGE = (255, 140, 0)
GEM_COLORS = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, ORANGE]

# Tetromino Definitions with rotations (using offsets)
TETROMINO_SHAPES = {
    'I': [ [(0, 0), (1, 0), (2, 0), (3, 0)], [(1, -1), (1, 0), (1, 1), (1, 2)] ],
    'O': [ [(0, 0), (1, 0), (0, 1), (1, 1)] ],
    'T': [ [(1, 0), (0, 1), (1, 1), (2, 1)], [(1, 0), (1, 1), (2, 1), (1, 2)], [(0, 1), (1, 1), (2, 1), (1, 2)], [(1, 0), (0, 1), (1, 1), (1, 2)] ],
    'L': [ [(0, 0), (0, 1), (0, 2), (1, 2)], [(0, 0), (1, 0), (2, 0), (0, 1)], [(0, 0), (1, 0), (1, 1), (1, 2)], [(2, 0), (0, 1), (1, 1), (2, 1)] ],
    'J': [ [(1, 0), (1, 1), (1, 2), (0, 2)], [(0, 0), (0, 1), (1, 1), (2, 1)], [(0, 0), (1, 0), (0, 1), (0, 2)], [(0, 0), (1, 0), (2, 0), (2, 1)] ]
}

class ScoreManager:
    def __init__(self):
        self.score = 0
        self.chain_multiplier = 1

    def add_line_clear(self, num_lines):
        self.score += BASE_LINE_CLEAR_POINTS * num_lines * self.chain_multiplier

    def add_match3_clear(self, group_size):
        self.score += BASE_MATCH3_POINTS * group_size * self.chain_multiplier

    def increase_chain(self):
        self.chain_multiplier += 1

    def reset_chain(self):
        self.chain_multiplier = 1

    def get_score(self):
        return self.score

class Tetromino:
    def __init__(self, shape):
        self.shape = shape
        self.rotations = TETROMINO_SHAPES[shape]
        self.rotation_index = 0
        self.blocks = self.rotations[self.rotation_index]
        self.x = GRID_COLS // 2 - 2
        self.y = 0
        self.color = random.choice(GEM_COLORS)

    def get_cells(self):
        return [(self.x + dx, self.y + dy) for dx, dy in self.blocks]

    def check_collision(self, grid, x, y, blocks):
        for bx, by in blocks:
            new_x = x + bx
            new_y = y + by
            if new_x < 0 or new_x >= GRID_COLS or new_y < 0 or new_y >= GRID_ROWS:
                return True
            if grid[new_y][new_x] is not None:
                return True
        return False

    def rotate(self, grid):
        next_index = (self.rotation_index + 1) % len(self.rotations)
        next_blocks = self.rotations[next_index]
        for dx, dy in [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]:
            if not self.check_collision(grid, self.x + dx, self.y + dy, next_blocks):
                self.x += dx
                self.y += dy
                self.rotation_index = next_index
                self.blocks = next_blocks
                break

    def move(self, dx, dy, grid):
        if not self.check_collision(grid, self.x + dx, self.y + dy, self.blocks):
            self.x += dx
            self.y += dy
            return True
        return False

    def lock_to_grid(self, grid):
        for bx, by in self.blocks:
            grid_y = self.y + by
            grid_x = self.x + bx
            if 0 <= grid_x < GRID_COLS and 0 <= grid_y < GRID_ROWS:
                grid[grid_y][grid_x] = self.color

class GridManager:
    def __init__(self):
        self.grid = [[None for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    def draw(self, surface):
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, GRAY, rect, 1)
                if self.grid[y][x]:
                    pygame.draw.rect(surface, self.grid[y][x], rect.inflate(-2, -2))

    def clear_full_rows(self, score_manager):
        cleared = 0
        new_grid = []
        for row in self.grid:
            if all(cell is not None for cell in row):
                cleared += 1
                new_grid.insert(0, [None for _ in range(GRID_COLS)])
            else:
                new_grid.append(row)
        self.grid = new_grid
        if cleared > 0:
            score_manager.add_line_clear(cleared)
        return cleared

    def _mark_horizontal_matches(self):
        to_clear = set()
        for row in range(GRID_ROWS):
            count = 1
            for col in range(1, GRID_COLS):
                if self.grid[row][col] is not None and self.grid[row][col] == self.grid[row][col - 1]:
                    count += 1
                else:
                    if count >= 3 and self.grid[row][col - 1] is not None:
                        for c in range(col - count, col):
                            to_clear.add((row, c))
                    count = 1
            if count >= 3 and self.grid[row][GRID_COLS - 1] is not None:
                for c in range(GRID_COLS - count, GRID_COLS):
                    to_clear.add((row, c))
        return to_clear

    def _mark_vertical_matches(self):
        to_clear = set()
        for col in range(GRID_COLS):
            count = 1
            for row in range(1, GRID_ROWS):
                if self.grid[row][col] is not None and self.grid[row][col] == self.grid[row - 1][col]:
                    count += 1
                else:
                    if count >= 3 and self.grid[row - 1][col] is not None:
                        for r in range(row - count, row):
                            to_clear.add((r, col))
                    count = 1
            if count >= 3 and self.grid[GRID_ROWS - 1][col] is not None:
                for r in range(GRID_ROWS - count, GRID_ROWS):
                    to_clear.add((r, col))
        return to_clear

    def clear_match3(self, score_manager):
        horizontal_matches = self._mark_horizontal_matches()
        vertical_matches = self._mark_vertical_matches()
        all_matches = horizontal_matches.union(vertical_matches)
        if all_matches:
            group_size = len(all_matches)
            score_manager.add_match3_clear(group_size)
            for r, c in all_matches:
                self.grid[r][c] = None
        return len(all_matches)

    def apply_gravity(self):
        for col in range(GRID_COLS):
            stack = [self.grid[row][col] for row in range(GRID_ROWS) if self.grid[row][col] is not None]
            for row in range(GRID_ROWS - 1, -1, -1):
                self.grid[row][col] = stack.pop() if stack else None

def execute_chain_reaction(grid_manager, score_manager):
    chain = 0
    while True:
        chain_occurred = False
        lines_cleared = grid_manager.clear_full_rows(score_manager)
        if lines_cleared > 0:
            chain_occurred = True
        gems_cleared = grid_manager.clear_match3(score_manager)
        if gems_cleared > 0:
            chain_occurred = True
        if not chain_occurred:
            break
        grid_manager.apply_gravity()
        chain += 1
        score_manager.increase_chain()
    score_manager.reset_chain()

def draw_tetromino(surface, tetromino):
    for dx, dy in tetromino.blocks:
        x = (tetromino.x + dx) * CELL_SIZE
        y = (tetromino.y + dy) * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, tetromino.color, rect.inflate(-2, -2))
        pygame.draw.rect(surface, GRAY, rect, 1)

def draw_next_tetromino(surface, tetromino, offset_x, offset_y):
    for dx, dy in tetromino.blocks:
        x = offset_x + dx * CELL_SIZE
        y = offset_y + dy * CELL_SIZE
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, tetromino.color, rect.inflate(-2, -2))
        pygame.draw.rect(surface, GRAY, rect, 1)

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tetris-Bejeweled Fusion")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 24)
        self.grid_manager = GridManager()
        self.score_manager = ScoreManager()
        self.current_piece = Tetromino(random.choice(list(TETROMINO_SHAPES.keys())))
        self.next_piece = Tetromino(random.choice(list(TETROMINO_SHAPES.keys())))
        self.drop_timer = 0
        self.drop_interval = 500  # milliseconds

    def run(self):
        while True:
            dt = self.clock.tick(FPS)
            self.handle_events()
            self.update(dt)
            self.render()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.current_piece.move(-1, 0, self.grid_manager.grid)
                elif event.key == pygame.K_RIGHT:
                    self.current_piece.move(1, 0, self.grid_manager.grid)
                elif event.key == pygame.K_DOWN:
                    self.current_piece.move(0, 1, self.grid_manager.grid)
                elif event.key == pygame.K_UP:
                    self.current_piece.rotate(self.grid_manager.grid)

    def update(self, dt):
        self.drop_timer += dt
        if self.drop_timer >= self.drop_interval:
            if not self.current_piece.move(0, 1, self.grid_manager.grid):
                self.current_piece.lock_to_grid(self.grid_manager.grid)
                execute_chain_reaction(self.grid_manager, self.score_manager)
                self.current_piece = self.next_piece
                self.next_piece = Tetromino(random.choice(list(TETROMINO_SHAPES.keys())))
                if self.current_piece.check_collision(self.grid_manager.grid, self.current_piece.x, self.current_piece.y, self.current_piece.blocks):
                    pygame.quit()
                    sys.exit()
            self.drop_timer = 0

    def render(self):
        self.screen.fill(BLACK)
        grid_surface = pygame.Surface((CELL_SIZE * GRID_COLS, CELL_SIZE * GRID_ROWS))
        grid_surface.fill(BLACK)
        self.grid_manager.draw(grid_surface)
        draw_tetromino(grid_surface, self.current_piece)
        self.screen.blit(grid_surface, (0, 0))
        panel_x = CELL_SIZE * GRID_COLS
        pygame.draw.rect(self.screen, GRAY, (panel_x, 0, SIDE_PANEL_WIDTH, WINDOW_HEIGHT))
        score_text = self.font.render("Score:", True, WHITE)
        self.screen.blit(score_text, (panel_x + 10, 20))
        score_val = self.font.render(str(self.score_manager.get_score()), True, WHITE)
        self.screen.blit(score_val, (panel_x + 10, 50))
        next_text = self.font.render("Next:", True, WHITE)
        self.screen.blit(next_text, (panel_x + 10, 100))
        draw_next_tetromino(self.screen, self.next_piece, panel_x + 10, 130)
        pygame.display.update()

if __name__ == "__main__":
    Game().run()
