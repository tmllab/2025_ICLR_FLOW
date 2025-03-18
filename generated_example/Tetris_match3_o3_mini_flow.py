

import sys, random, pygame, time

# -------------------- Constants & Global Settings --------------------
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 500
GRID_COLS = 10
GRID_ROWS = 20
CELL_SIZE = 24
GRID_ORIGIN = (20, 20)
NEXT_ORIGIN = (300, 50)
SCORE_POS = (300, 250)
BG_COLOR = (30, 30, 30)
BORDER_COLOR = (50, 50, 50)
TEXT_COLOR = (255, 255, 255)
FALL_DELAY = 500  # milliseconds delay for tetromino fall

# Tetromino Shapes (list of (dx, dy) offsets relative to a pivot)
SHAPES = {
    "I": [(-1, 0), (0, 0), (1, 0), (2, 0)],
    "O": [(0, 0), (1, 0), (0, 1), (1, 1)],
    "T": [(-1, 0), (0, 0), (1, 0), (0, 1)],
    "S": [(0, 0), (1, 0), (-1, 1), (0, 1)],
    "Z": [(-1, 0), (0, 0), (0, 1), (1, 1)],
    "J": [(-1, 0), (0, 0), (1, 0), (-1, 1)],
    "L": [(-1, 0), (0, 0), (1, 0), (1, 1)]
}

# Mapping tetromino shape to gem color strings
COLOR_MAPPING = {
    "I": "cyan",
    "O": "yellow",
    "T": "purple",
    "S": "green",
    "Z": "red",
    "J": "blue",
    "L": "orange"
}

# Map color names to RGB values (for display in grid)
RGB_GEMS = {
    "cyan":   (0, 255, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "green":  (0, 255, 0),
    "red":    (255, 0, 0),
    "blue":   (0, 0, 255),
    "orange": (255, 165, 0)
}

# -------------------- Game Core Modules --------------------
class GemTransformation:
    def transform_to_gems(self, tetromino, grid):
        # Convert locked tetromino blocks into colored gems using the mapping
        gem_color = COLOR_MAPPING[tetromino["shape"]]
        for (x, y) in tetromino["blocks"]:
            if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
                grid[y][x] = gem_color
        return grid

class TetrisMechanics:
    def __init__(self, grid_width=GRID_COLS, grid_height=GRID_ROWS):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.grid = [[None for _ in range(grid_width)] for _ in range(grid_height)]
        self.current_tetromino = None
        self.gem_transformer = GemTransformation()

    def spawn_tetromino(self):
        shape = random.choice(list(SHAPES.keys()))
        pivot_x = self.grid_width // 2
        pivot_y = 0
        blocks = []
        for offset in SHAPES[shape]:
            x = pivot_x + offset[0]
            y = pivot_y + offset[1]
            blocks.append((x, y))
        self.current_tetromino = {
            "shape": shape,
            "pivot": (pivot_x, pivot_y),
            "blocks": blocks
        }

    def check_collision(self, blocks):
        for (x, y) in blocks:
            if x < 0 or x >= self.grid_width or y >= self.grid_height:
                return True
            if y >= 0 and self.grid[y][x] is not None:
                return True
        return False

    def move_tetromino(self, direction):
        if self.current_tetromino is None:
            return
        dx, dy = 0, 0
        if direction == 'left':
            dx = -1
        elif direction == 'right':
            dx = 1
        elif direction == 'down':
            dy = 1
        new_blocks = [(x + dx, y + dy) for (x, y) in self.current_tetromino["blocks"]]
        if not self.check_collision(new_blocks):
            self.current_tetromino["blocks"] = new_blocks
            pivot_x, pivot_y = self.current_tetromino["pivot"]
            self.current_tetromino["pivot"] = (pivot_x + dx, pivot_y + dy)
        elif direction == 'down':
            self.lock_piece()

    def rotate_tetromino(self):
        if self.current_tetromino is None:
            return
        pivot = self.current_tetromino["pivot"]
        new_blocks = []
        for (x, y) in self.current_tetromino["blocks"]:
            rel_x = x - pivot[0]
            rel_y = y - pivot[1]
            # Rotate clockwise: (x, y) -> (y, -x)
            new_x = pivot[0] + rel_y
            new_y = pivot[1] - rel_x
            new_blocks.append((new_x, new_y))
        if not self.check_collision(new_blocks):
            self.current_tetromino["blocks"] = new_blocks

    def lock_piece(self):
        if self.current_tetromino is None:
            return
        self.gem_transformer.transform_to_gems(self.current_tetromino, self.grid)
        self.current_tetromino = None

# -- Match-3 Clearing and Tetris Line Clear Logic --
class Match3Controller:
    def __init__(self, min_match=3):
        self.min_match = min_match

    def detect_matches(self, gem_grid):
        overall_matches = set()
        rows = len(gem_grid)
        cols = len(gem_grid[0])
        # Detect horizontal matches
        for r in range(rows):
            count = 1
            for c in range(1, cols):
                if gem_grid[r][c] is not None and gem_grid[r][c] == gem_grid[r][c - 1]:
                    count += 1
                else:
                    if count >= self.min_match:
                        positions = {(r, k) for k in range(c - count, c)}
                        overall_matches.update(positions)
                    count = 1
            if count >= self.min_match:
                positions = {(r, k) for k in range(cols - count, cols)}
                overall_matches.update(positions)
        # Detect vertical matches
        for c in range(cols):
            count = 1
            for r in range(1, rows):
                if gem_grid[r][c] is not None and gem_grid[r][c] == gem_grid[r - 1][c]:
                    count += 1
                else:
                    if count >= self.min_match:
                        positions = {(k, c) for k in range(r - count, r)}
                        overall_matches.update(positions)
                    count = 1
            if count >= self.min_match:
                positions = {(k, c) for k in range(rows - count, rows)}
                overall_matches.update(positions)
        # Diagonal detection (down-right)
        for r in range(rows):
            for c in range(cols):
                positions = [(r, c)]
                rr, cc = r + 1, c + 1
                while rr < rows and cc < cols and gem_grid[rr][cc] is not None and gem_grid[rr][cc] == gem_grid[r][c]:
                    positions.append((rr, cc))
                    rr += 1
                    cc += 1
                if len(positions) >= self.min_match:
                    overall_matches.update(positions)
        # Diagonal detection (down-left)
        for r in range(rows):
            for c in range(cols):
                positions = [(r, c)]
                rr, cc = r + 1, c - 1
                while rr < rows and cc >= 0 and gem_grid[rr][cc] is not None and gem_grid[rr][cc] == gem_grid[r][c]:
                    positions.append((rr, cc))
                    rr += 1
                    cc -= 1
                if len(positions) >= self.min_match:
                    overall_matches.update(positions)
        return overall_matches

    def clear_matches(self, gem_grid, matched_positions):
        cleared_count = 0
        for (r, c) in matched_positions:
            if gem_grid[r][c] is not None:
                gem_grid[r][c] = None
                cleared_count += 1
        return cleared_count

    def drop_gems(self, gem_grid):
        rows = len(gem_grid)
        cols = len(gem_grid[0])
        for c in range(cols):
            empty_slots = []
            for r in range(rows - 1, -1, -1):
                if gem_grid[r][c] is None:
                    empty_slots.append(r)
                elif empty_slots:
                    empty_r = empty_slots.pop(0)
                    gem_grid[empty_r][c] = gem_grid[r][c]
                    gem_grid[r][c] = None
                    empty_slots.append(r)
        return gem_grid

    def trigger_chain_reaction(self, gem_grid):
        chain_multiplier = 1
        total_bonus = 0
        while True:
            matches = self.detect_matches(gem_grid)
            if not matches:
                break
            cleared = self.clear_matches(gem_grid, matches)
            bonus = cleared * chain_multiplier * 10
            total_bonus += bonus
            gem_grid = self.drop_gems(gem_grid)
            chain_multiplier += 1
        return gem_grid, total_bonus

def clear_tetris_lines(grid):
    rows = len(grid)
    cols = len(grid[0])
    lines_cleared = 0
    new_grid = []
    for row in grid:
        if all(cell is not None for cell in row):
            lines_cleared += 1
        else:
            new_grid.append(row)
    while len(new_grid) < rows:
        new_grid.insert(0, [None for _ in range(cols)])
    return new_grid, lines_cleared

class ScoreManager:
    def __init__(self):
        self.total_score = 0

    def update_tetris_score(self, lines_cleared):
        self.total_score += lines_cleared * 100

    def update_match3_score(self, gems_cleared, bonus):
        self.total_score += gems_cleared * 5 + bonus

    def get_total_score(self):
        return self.total_score

# -------------------- GUI Module (Using Pygame) --------------------
class GameGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tetris-Bejeweled Fusion")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 20)

    def draw_grid(self, gem_grid):
        # Draw the game grid with each cell as a colored block (or black if empty)
        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                cell_value = gem_grid[row][col]
                if cell_value is None:
                    color = (0, 0, 0)
                else:
                    color = RGB_GEMS.get(cell_value, (255, 255, 255))
                rect = pygame.Rect(GRID_ORIGIN[0] + col * CELL_SIZE,
                                   GRID_ORIGIN[1] + row * CELL_SIZE,
                                   CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, BORDER_COLOR, rect, 1)

    def draw_score(self, score):
        score_text = self.font.render("Score: " + str(score), True, TEXT_COLOR)
        self.screen.blit(score_text, SCORE_POS)

    def draw_next_tetromino(self, tetromino):
        # Represent the tetromino as a simple matrix using 1 for blocks and 0 for empty.
        # First, convert tetromino blocks to a minimal matrix by finding bounds.
        matrix = piece_to_matrix(tetromino)
        block_color = (200, 200, 200)
        for row_idx, row in enumerate(matrix):
            for col_idx, cell in enumerate(row):
                if cell:
                    rect = pygame.Rect(NEXT_ORIGIN[0] + col_idx * CELL_SIZE,
                                       NEXT_ORIGIN[1] + row_idx * CELL_SIZE,
                                       CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, block_color, rect)
                    pygame.draw.rect(self.screen, BORDER_COLOR, rect, 1)
        header = self.font.render("Next", True, TEXT_COLOR)
        self.screen.blit(header, (NEXT_ORIGIN[0], NEXT_ORIGIN[1] - 25))

    def update_display(self):
        pygame.display.flip()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

def piece_to_matrix(piece):
    # Convert tetromino block positions into a minimal 2D matrix representation.
    coords = piece["blocks"]
    if not coords:
        return [[0]]
    xs = [x for x,y in coords]
    ys = [y for x,y in coords]
    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    matrix = [[0 for _ in range(width)] for _ in range(height)]
    for (x,y) in coords:
        mx = x - min_x
        my = y - min_y
        matrix[my][mx] = 1
    return matrix

# -------------------- Integration: Main Game Loop --------------------
def main():
    # Instantiate core modules
    mechanics = TetrisMechanics()
    matcher = Match3Controller()
    scorer = ScoreManager()
    gui = GameGUI()

    # Spawn first tetromino and set a next-piece preview
    mechanics.spawn_tetromino()
    next_piece = None
    if mechanics.current_tetromino:
        # Prepare a preview piece by generating a new tetromino without locking it immediately.
        next_piece = {}
        shape = random.choice(list(SHAPES.keys()))
        pivot = (GRID_COLS // 2, 0)
        blocks = [(pivot[0] + dx, pivot[1] + dy) for (dx,dy) in SHAPES[shape]]
        next_piece = {"shape": shape, "pivot": pivot, "blocks": blocks}

    last_fall = pygame.time.get_ticks()
    game_over = False

    while not game_over:
        gui.handle_input()

        # --- Handle user input for tetromino movement ---
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            mechanics.move_tetromino("left")
        if keys[pygame.K_RIGHT]:
            mechanics.move_tetromino("right")
        if keys[pygame.K_DOWN]:
            mechanics.move_tetromino("down")
        if keys[pygame.K_UP]:
            mechanics.rotate_tetromino()

        # --- Gravity: Automatic tetromino move down ---
        now = pygame.time.get_ticks()
        if now - last_fall > FALL_DELAY:
            mechanics.move_tetromino("down")
            last_fall = now

        # --- Check if tetromino has locked and process grid clearing ---
        if mechanics.current_tetromino is None:
            # First, clear any full lines (Tetris mechanic)
            mechanics.grid, lines = clear_tetris_lines(mechanics.grid)
            scorer.update_tetris_score(lines)
            # Then, check and process gem match-3 clearing with chain reactions
            before = sum(1 for row in mechanics.grid for cell in row if cell is not None)
            mechanics.grid, bonus = matcher.trigger_chain_reaction(mechanics.grid)
            after = sum(1 for row in mechanics.grid for cell in row if cell is not None)
            cleared_gems = before - after
            scorer.update_match3_score(cleared_gems, bonus)
            # Update current piece (use the pre-generated next_piece) and spawn a new next preview
            mechanics.spawn_tetromino()
            current = mechanics.current_tetromino
            next_piece = None
            shape = random.choice(list(SHAPES.keys()))
            pivot = (GRID_COLS // 2, 0)
            blocks = [(pivot[0] + dx, pivot[1] + dy) for (dx,dy) in SHAPES[shape]]
            next_piece = {"shape": shape, "pivot": pivot, "blocks": blocks}
            # Check for game over: if any block of the new tetromino collides with existing gems
            for (x, y) in current["blocks"]:
                if y < 0 or mechanics.grid[y][x] is not None:
                    game_over = True
                    break

        # --- GUI Rendering ---
        gui.screen.fill(BG_COLOR)
        gui.draw_grid(mechanics.grid)
        # Draw current falling tetromino if available
        if mechanics.current_tetromino:
            # Draw the tetromino by rendering each block with its corresponding gem color
            gem_color = RGB_GEMS[COLOR_MAPPING[mechanics.current_tetromino["shape"]]]
            for (x, y) in mechanics.current_tetromino["blocks"]:
                if y >= 0:
                    rect = pygame.Rect(GRID_ORIGIN[0] + x * CELL_SIZE, GRID_ORIGIN[1] + y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(gui.screen, gem_color, rect)
                    pygame.draw.rect(gui.screen, BORDER_COLOR, rect, 1)
        gui.draw_score(scorer.get_total_score())
        if next_piece:
            gui.draw_next_tetromino(next_piece)
        gui.update_display()
        gui.clock.tick(60)

    # Game Over — show final score for a short moment then exit
    print("Game Over! Final Score:", scorer.get_total_score())
    time.sleep(2)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
