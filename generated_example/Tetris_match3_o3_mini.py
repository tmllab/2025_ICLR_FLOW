import pygame
import random
import sys
import copy

# Game constants
GRID_WIDTH, GRID_HEIGHT = 10, 20
CELL_SIZE = 30
FPS = 60
ANIM_SPEED = 5  # pixels per frame for falling animation

# Colors (R, G, B)
BLACK    = (0, 0, 0)
WHITE    = (255, 255, 255)
GRAY     = (128, 128, 128)
RED      = (255, 0, 0)
GREEN    = (0, 255, 0)
BLUE     = (0, 0, 255)
YELLOW   = (255, 255, 0)
CYAN     = (0, 255, 255)
MAGENTA  = (255, 0, 255)
ORANGE   = (255, 165, 0)

GEM_COLORS = [RED, GREEN, BLUE, YELLOW, CYAN, MAGENTA, ORANGE]

# Tetromino shapes (using 4x4 grid representation)
TETROMINOES = {
    'I': [[0, 0, 0, 0],
          [1, 1, 1, 1],
          [0, 0, 0, 0],
          [0, 0, 0, 0]],
    'O': [[1, 1],
          [1, 1]],
    'T': [[0, 1, 0],
          [1, 1, 1],
          [0, 0, 0]],
    'S': [[0, 1, 1],
          [1, 1, 0],
          [0, 0, 0]],
    'Z': [[1, 1, 0],
          [0, 1, 1],
          [0, 0, 0]],
    'J': [[1, 0, 0],
          [1, 1, 1],
          [0, 0, 0]],
    'L': [[0, 0, 1],
          [1, 1, 1],
          [0, 0, 0]]
}

def rotate(shape):
    # Rotate the matrix clockwise
    return [list(row) for row in zip(*shape[::-1])]

class Tetromino:
    def __init__(self, shape_id=None):
        if shape_id is None:
            self.shape_id = random.choice(list(TETROMINOES.keys()))
        else:
            self.shape_id = shape_id
        self.shape = copy.deepcopy(TETROMINOES[self.shape_id])
        self.x = GRID_WIDTH // 2 - len(self.shape[0]) // 2
        self.y = 0
        # Assign a random gem color for the tetromino
        self.color = random.choice(GEM_COLORS)
    
    def rotate(self):
        self.shape = rotate(self.shape)
    
    @property
    def width(self):
        return len(self.shape[0])
    
    @property
    def height(self):
        return len(self.shape)

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((CELL_SIZE * (GRID_WIDTH + 6), CELL_SIZE * GRID_HEIGHT))
        pygame.display.set_caption("Tetris-Bejeweled Fusion")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.reset()

    def reset(self):
        self.grid = [[None for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
        self.score = 0
        self.current_piece = Tetromino()
        self.next_piece = Tetromino()
        self.drop_timer = 0
        self.drop_interval = 500  # milliseconds
        self.game_over = False

    def valid_position(self, tetromino, adj_x=0, adj_y=0):
        for y, row in enumerate(tetromino.shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = tetromino.x + x + adj_x
                    new_y = tetromino.y + y + adj_y
                    if new_x < 0 or new_x >= GRID_WIDTH or new_y >= GRID_HEIGHT:
                        return False
                    if new_y >= 0 and self.grid[new_y][new_x] is not None:
                        return False
        return True

    def lock_piece(self, tetromino):
        # When locking, convert each block into a gem with its color.
        for y, row in enumerate(tetromino.shape):
            for x, cell in enumerate(row):
                if cell:
                    grid_x = tetromino.x + x
                    grid_y = tetromino.y + y
                    if grid_y < 0:
                        self.game_over = True
                        return
                    self.grid[grid_y][grid_x] = tetromino.color
        # After locking, process clearing
        self.process_clears()

    def process_clears(self):
        lines_cleared = self.clear_full_lines()
        match_cleared = self.clear_matches()
        if lines_cleared or match_cleared:
            self.apply_gravity()
            # Check for chain reactions
            self.process_clears()
    
    def clear_full_lines(self):
        full_lines = [i for i, row in enumerate(self.grid) if all(cell is not None for cell in row)]
        if full_lines:
            for i in full_lines:
                # Simple animation: flash the row white briefly (not fully implemented for brevity)
                self.score += 100  # score for line clear
                del self.grid[i]
                self.grid.insert(0, [None for _ in range(GRID_WIDTH)])
            return True
        return False

    def clear_matches(self):
        """Perform a flood-fill to find connected gems of the same color.
           If a group of 3 or more is found, clear them."""
        to_clear = set()
        visited = [[False] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] is not None and not visited[y][x]:
                    color = self.grid[y][x]
                    group = self.flood_fill(x, y, color, visited)
                    if len(group) >= 3:
                        to_clear |= group
        for (x, y) in to_clear:
            self.grid[y][x] = None
            self.score += 50  # score for match clear
        return bool(to_clear)

    def flood_fill(self, x, y, color, visited):
        stack = [(x, y)]
        group = set()
        while stack:
            cx, cy = stack.pop()
            if (0 <= cx < GRID_WIDTH and 0 <= cy < GRID_HEIGHT and 
                not visited[cy][cx] and self.grid[cy][cx] == color):
                visited[cy][cx] = True
                group.add((cx, cy))
                # Check four directions (up, down, left, right)
                stack.extend([(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)])
        return group

    def apply_gravity(self):
        # Make gems fall into empty spaces
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT - 2, -1, -1):
                if self.grid[y][x] is not None:
                    fall_y = y
                    while fall_y + 1 < GRID_HEIGHT and self.grid[fall_y + 1][x] is None:
                        fall_y += 1
                    if fall_y != y:
                        self.grid[fall_y][x] = self.grid[y][x]
                        self.grid[y][x] = None

    def draw_grid(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, GRAY, rect, 1)
                if self.grid[y][x]:
                    pygame.draw.rect(self.screen, self.grid[y][x], rect.inflate(-2, -2))

    def draw_piece(self, tetromino, offset_x=0, offset_y=0):
        for y, row in enumerate(tetromino.shape):
            for x, cell in enumerate(row):
                if cell:
                    px = (tetromino.x + x + offset_x) * CELL_SIZE
                    py = (tetromino.y + y + offset_y) * CELL_SIZE
                    rect = pygame.Rect(px, py, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, tetromino.color, rect.inflate(-2, -2))

    def draw_preview(self):
        # Draw next piece in a box on the right
        preview_x = GRID_WIDTH * CELL_SIZE + 20
        preview_y = 50
        preview_surface = pygame.Surface((CELL_SIZE * 4, CELL_SIZE * 4))
        preview_surface.fill(BLACK)
        shape = self.next_piece.shape
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(preview_surface, self.next_piece.color, rect.inflate(-2, -2))
        self.screen.blit(preview_surface, (preview_x, preview_y))
        label = self.font.render("Next:", True, WHITE)
        self.screen.blit(label, (preview_x, preview_y - 30))

    def draw_score(self):
        score_surf = self.font.render(f"Score: {self.score}", True, WHITE)
        self.screen.blit(score_surf, (GRID_WIDTH * CELL_SIZE + 20, 10))

    def run(self):
        last_drop = pygame.time.get_ticks()
        while not self.game_over:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # Handle keyboard input
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and self.valid_position(self.current_piece, adj_x=-1):
                        self.current_piece.x -= 1
                    elif event.key == pygame.K_RIGHT and self.valid_position(self.current_piece, adj_x=1):
                        self.current_piece.x += 1
                    elif event.key == pygame.K_DOWN and self.valid_position(self.current_piece, adj_y=1):
                        self.current_piece.y += 1
                    elif event.key == pygame.K_UP:
                        # Rotate and check for valid position
                        old_shape = copy.deepcopy(self.current_piece.shape)
                        self.current_piece.rotate()
                        if not self.valid_position(self.current_piece):
                            self.current_piece.shape = old_shape

            # Automatic drop
            now = pygame.time.get_ticks()
            if now - last_drop > self.drop_interval:
                if self.valid_position(self.current_piece, adj_y=1):
                    self.current_piece.y += 1
                else:
                    self.lock_piece(self.current_piece)
                    self.current_piece = self.next_piece
                    self.next_piece = Tetromino()
                    if not self.valid_position(self.current_piece):
                        self.game_over = True
                last_drop = now

            # Draw everything
            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_piece(self.current_piece)
            self.draw_preview()
            self.draw_score()
            pygame.display.update()

        # Game Over Screen
        self.screen.fill(BLACK)
        go_text = self.font.render("Game Over", True, WHITE)
        score_text = self.font.render(f"Final Score: {self.score}", True, WHITE)
        self.screen.blit(go_text, (CELL_SIZE * 2, CELL_SIZE * 8))
        self.screen.blit(score_text, (CELL_SIZE * 2, CELL_SIZE * 10))
        pygame.display.update()
        pygame.time.wait(3000)
        pygame.quit()

if __name__ == "__main__":
    Game().run()
