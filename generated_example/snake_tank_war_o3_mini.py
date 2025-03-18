import pygame
import random
import math
import sys

# ----------------------------
# Constants and Configuration
# ----------------------------
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
GRID_SIZE = 20  # size of grid cell (pixels)
FPS = 15

# Colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY  = (50, 50, 50)
RED   = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE  = (0, 0, 255)
YELLOW= (255, 255, 0)
ORANGE= (255, 165, 0)

# ----------------------------
# Helper Functions
# ----------------------------
def draw_text(surface, text, pos, color=WHITE, size=20):
    font = pygame.font.SysFont("Arial", size)
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, pos)

def grid_to_pixels(pos):
    """Convert grid coordinates (col, row) to pixel coordinates."""
    return (pos[0]*GRID_SIZE, pos[1]*GRID_SIZE)

def pixel_to_grid(pos):
    """Convert pixel coordinates to grid cell coordinates."""
    return (pos[0] // GRID_SIZE, pos[1] // GRID_SIZE)

def angle_to_vector(angle):
    """Convert an angle (in degrees) to a unit vector (dx, dy)."""
    rad = math.radians(angle)
    return (math.cos(rad), -math.sin(rad))

# ----------------------------
# Game Object Classes
# ----------------------------
class Tank:
    def __init__(self, x, y, angle=0, color=GREEN):
        self.x = x
        self.y = y
        self.angle = angle  # in degrees; 0 = right
        self.color = color
        self.health = 100
        self.fuel = 100
        self.score = 0
        self.speed = 1  # moves 1 grid per frame
        self.size = GRID_SIZE
        self.rect = pygame.Rect(self.x * GRID_SIZE, self.y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    
    def update_rect(self):
        self.rect.topleft = grid_to_pixels((self.x, self.y))
    
    def move_forward(self):
        dx, dy = angle_to_vector(self.angle)
        self.x += int(round(dx * self.speed))
        self.y += int(round(dy * self.speed))
        self.update_rect()
    
    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)
        # Draw barrel to indicate direction
        barrel_length = GRID_SIZE//2
        dx, dy = angle_to_vector(self.angle)
        start = self.rect.center
        end = (start[0] + dx * barrel_length, start[1] + dy * barrel_length)
        pygame.draw.line(surface, BLACK, start, end, 3)
    
class PlayerTank(Tank):
    def __init__(self, x, y, angle=0):
        super().__init__(x, y, angle, color=GREEN)
        self.fuel_trail = []  # list of grid positions
        self.trail_length = 30  # maximum length of fuel trail
        self.ammo = 10

    def update_trail(self):
        # Append current head position to fuel trail
        pos = (self.x, self.y)
        if not self.fuel_trail or self.fuel_trail[-1] != pos:
            self.fuel_trail.append(pos)
        # Limit the trail length
        if len(self.fuel_trail) > self.trail_length:
            self.fuel_trail.pop(0)
    
    def draw(self, surface):
        # Draw fuel trail (hazardous for self-collision)
        for pos in self.fuel_trail:
            rect = pygame.Rect(grid_to_pixels(pos), (GRID_SIZE, GRID_SIZE))
            pygame.draw.rect(surface, ORANGE, rect)
        super().draw(surface)

class EnemyTank(Tank):
    def __init__(self, x, y, angle=180):
        super().__init__(x, y, angle, color=RED)
        self.ammo = 5
        self.shoot_delay = 0

    def update_ai(self, player):
        # Simple AI: aim towards player and periodically shoot.
        dx = player.x - self.x
        dy = player.y - self.y
        target_angle = math.degrees(math.atan2(-dy, dx))
        # Smooth rotation: gradually adjust angle
        angle_diff = (target_angle - self.angle + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        self.angle += max(-5, min(5, angle_diff))  # adjust up to 5 degrees per frame

        # Move slowly towards player if far away
        dist = math.hypot(dx, dy)
        if dist > 5:
            self.move_forward()
        
        # Shooting delay countdown
        if self.shoot_delay > 0:
            self.shoot_delay -= 1

    def can_shoot(self):
        return self.shoot_delay == 0 and self.ammo > 0

    def reset_shoot_delay(self):
        self.shoot_delay = FPS  # can shoot once per second (approximately)

class Bullet:
    def __init__(self, x, y, angle, owner, speed=2):
        self.x = x
        self.y = y
        self.angle = angle
        self.owner = owner  # reference to the Tank that shot it
        self.speed = speed
        self.size = GRID_SIZE // 2
        self.rect = pygame.Rect(self.x * GRID_SIZE, self.y * GRID_SIZE, self.size, self.size)
    
    def update_rect(self):
        self.rect.topleft = grid_to_pixels((self.x, self.y))
    
    def move(self):
        dx, dy = angle_to_vector(self.angle)
        self.x += dx * self.speed
        self.y += dy * self.speed
        self.update_rect()
    
    def draw(self, surface):
        pygame.draw.rect(surface, YELLOW, self.rect)

class Resource:
    def __init__(self, x, y, kind='fuel'):
        self.x = x
        self.y = y
        self.kind = kind  # either 'fuel' or 'ammo'
        self.size = GRID_SIZE
        self.rect = pygame.Rect(self.x * GRID_SIZE, self.y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
        self.color = BLUE if kind == 'fuel' else WHITE
    
    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = GRID_SIZE
        self.rect = pygame.Rect(self.x * GRID_SIZE, self.y * GRID_SIZE, GRID_SIZE, GRID_SIZE)
    
    def draw(self, surface):
        pygame.draw.rect(surface, GRAY, self.rect)

# ----------------------------
# Main Game Class
# ----------------------------
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Battlefield Game: Snake Tank War")
        self.clock = pygame.time.Clock()
        self.running = True

        # Calculate grid dimensions
        self.cols = SCREEN_WIDTH // GRID_SIZE
        self.rows = SCREEN_HEIGHT // GRID_SIZE

        # Initialize game agents
        self.player = PlayerTank(self.cols // 4, self.rows // 2, angle=0)
        self.enemy = EnemyTank(self.cols * 3 // 4, self.rows // 2, angle=180)
        self.bullets = []
        self.obstacles = self.create_obstacles(20)
        self.resources = self.create_resources(10)

    def create_obstacles(self, count):
        obstacles = []
        for _ in range(count):
            x = random.randint(0, self.cols - 1)
            y = random.randint(0, self.rows - 1)
            obstacles.append(Obstacle(x, y))
        return obstacles

    def create_resources(self, count):
        resources = []
        for _ in range(count):
            x = random.randint(0, self.cols - 1)
            y = random.randint(0, self.rows - 1)
            kind = random.choice(['fuel', 'ammo'])
            resources.append(Resource(x, y, kind))
        return resources

    def run(self):
        while self.running:
            self.clock.tick(FPS)
            self.handle_events()
            self.update_game_state()
            self.render()
        pygame.quit()
        sys.exit()

    def handle_events(self):
        # Handle quit and key events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
        
        # Player controls: left/right turning using arrow keys.
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.player.angle = (self.player.angle + 10) % 360
        if keys[pygame.K_RIGHT]:
            self.player.angle = (self.player.angle - 10) % 360
        # Player shooting (space bar) if ammo available.
        if keys[pygame.K_SPACE] and self.player.ammo > 0:
            self.bullets.append(Bullet(self.player.x, self.player.y, self.player.angle, owner='player'))
            self.player.ammo -= 1

    def update_game_state(self):
        # Move player tank forward
        self.player.move_forward()
        self.player.update_trail()
        self.player.fuel = max(0, self.player.fuel - 0.1)  # decrease fuel gradually

        # Check player boundaries
        if not (0 <= self.player.x < self.cols and 0 <= self.player.y < self.rows):
            self.game_over("Player left the battlefield!")
        
        # Check collision with obstacles and own fuel trail
        if self.check_collision(self.player, self.obstacles) or self.check_self_collision():
            self.game_over("Player collided with an obstacle or own trail!")

        # Enemy AI updates and shooting
        self.enemy.update_ai(self.player)
        if self.enemy.can_shoot():
            self.bullets.append(Bullet(self.enemy.x, self.enemy.y, self.enemy.angle, owner='enemy'))
            self.enemy.ammo -= 1
            self.enemy.reset_shoot_delay()

        # Move bullets and check for collisions
        for bullet in self.bullets[:]:
            bullet.move()
            # Remove bullets that leave the battlefield boundaries.
            if not (0 <= bullet.x < self.cols and 0 <= bullet.y < self.rows):
                self.bullets.remove(bullet)
                continue

            # Check bullet collisions with tanks or obstacles.
            if bullet.owner == 'player':
                if self.rect_collision(bullet.rect, self.enemy.rect):
                    self.enemy.health -= 25
                    self.bullets.remove(bullet)
                    if self.enemy.health <= 0:
                        self.player.score += 100
                        # Respawn enemy
                        self.enemy = EnemyTank(self.cols * 3 // 4, random.randint(0, self.rows - 1), angle=180)
            elif bullet.owner == 'enemy':
                if self.rect_collision(bullet.rect, self.player.rect):
                    self.player.health -= 20
                    self.bullets.remove(bullet)
                    if self.player.health <= 0:
                        self.game_over("Player was destroyed by enemy fire!")
        
        # Check for resource collection
        for resource in self.resources[:]:
            if self.rect_collision(self.player.rect, resource.rect):
                if resource.kind == 'fuel':
                    self.player.fuel = min(100, self.player.fuel + 30)
                elif resource.kind == 'ammo':
                    self.player.ammo += 5
                self.player.score += 10
                self.resources.remove(resource)
                # Spawn a new resource to keep the game dynamic.
                self.resources.append(self.spawn_resource())

    def spawn_resource(self):
        x = random.randint(0, self.cols - 1)
        y = random.randint(0, self.rows - 1)
        kind = random.choice(['fuel', 'ammo'])
        return Resource(x, y, kind)

    def rect_collision(self, rect1, rect2):
        return rect1.colliderect(rect2)

    def check_collision(self, tank, objects):
        for obj in objects:
            if tank.rect.colliderect(obj.rect):
                return True
        return False

    def check_self_collision(self):
        # Check if the player's head collides with its fuel trail (excluding the latest few positions to avoid immediate collision)
        head = (self.player.x, self.player.y)
        if head in self.player.fuel_trail[:-5]:
            return True
        return False

    def render(self):
        self.screen.fill(BLACK)
        # Draw grid (optional for visual style)
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (SCREEN_WIDTH, y))
        
        # Draw game objects
        for obstacle in self.obstacles:
            obstacle.draw(self.screen)
        for resource in self.resources:
            resource.draw(self.screen)
        self.player.draw(self.screen)
        self.enemy.draw(self.screen)
        for bullet in self.bullets:
            bullet.draw(self.screen)
        
        # Draw HUD
        hud_text = f"Health: {int(self.player.health)}  Fuel: {int(self.player.fuel)}  Ammo: {self.player.ammo}  Score: {self.player.score}"
        draw_text(self.screen, hud_text, (10, 10), color=WHITE, size=20)
        
        pygame.display.flip()

    def game_over(self, message):
        print(message)
        # Simple game over screen
        self.screen.fill(BLACK)
        draw_text(self.screen, "GAME OVER", (SCREEN_WIDTH//2 - 80, SCREEN_HEIGHT//2 - 30), color=RED, size=40)
        draw_text(self.screen, message, (SCREEN_WIDTH//2 - 150, SCREEN_HEIGHT//2 + 10), color=WHITE, size=25)
        pygame.display.flip()
        pygame.time.delay(3000)
        self.running = False

# ----------------------------
# Run the Game
# ----------------------------
if __name__ == "__main__":
    game = Game()
    game.run()
