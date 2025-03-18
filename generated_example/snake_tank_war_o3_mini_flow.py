
#!/usr/bin/env python3
"""
Grid Battlefield Game
----------------------
A multi-agent dynamic grid-based battlefield game that blends continuous tank movement (with fuel trail hazard)
and strategic combat (enemy AI, shooting mechanics) with a real-time HUD.

Controls:
  - Left Arrow: steer the tank to the left
  - Right Arrow: steer the tank to the right
  - Spacebar: fire a bullet

Run the game normally:
    python game.py

Run integration tests:
    python game.py test
------------------------------------------------------------
"""

import pygame, sys, random, math

# Initialize pygame
pygame.init()

# Global Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CELL_SIZE = 20
GRID_WIDTH = SCREEN_WIDTH // CELL_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // CELL_SIZE
FPS = 10

# Colors
BLACK    = (0, 0, 0)
WHITE    = (255, 255, 255)
GREEN    = (0, 200, 0)
RED      = (200, 0, 0)
BLUE     = (0, 0, 200)
YELLOW   = (200, 200, 0)
GRAY     = (100, 100, 100)
HUD_BG_COLOR = (50, 50, 50)

# Directions
DIR_UP    = (0, -1)
DIR_DOWN  = (0, 1)
DIR_LEFT  = (-1, 0)
DIR_RIGHT = (1, 0)

#----------------------------------
# Agent and Object Classes
#----------------------------------

class PlayerTank:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction = DIR_RIGHT
        self.fuel_trail = [(x, y)]
        self.trail_length = 5
        self.health = 100
        self.fuel = 100
        self.score = 0

    def update(self):
        # Move in the current direction and extend the fuel trail
        self.x += self.direction[0]
        self.y += self.direction[1]
        self.fuel_trail.append((self.x, self.y))
        if len(self.fuel_trail) > self.trail_length:
            self.fuel_trail.pop(0)

    def steer(self, turn):
        if turn == "LEFT":
            if self.direction == DIR_RIGHT:    self.direction = DIR_UP
            elif self.direction == DIR_UP:     self.direction = DIR_LEFT
            elif self.direction == DIR_LEFT:   self.direction = DIR_DOWN
            elif self.direction == DIR_DOWN:   self.direction = DIR_RIGHT
        elif turn == "RIGHT":
            if self.direction == DIR_RIGHT:    self.direction = DIR_DOWN
            elif self.direction == DIR_DOWN:   self.direction = DIR_LEFT
            elif self.direction == DIR_LEFT:   self.direction = DIR_UP
            elif self.direction == DIR_UP:     self.direction = DIR_RIGHT

    def check_self_collision(self):
        # Self collision if any position repeats in the fuel trail
        return len(self.fuel_trail) != len(set(self.fuel_trail))

    def draw(self, surface):
        # Draw the fuel trail (as yellow cells)
        for pos in self.fuel_trail:
            rect = pygame.Rect(pos[0]*CELL_SIZE, pos[1]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(surface, YELLOW, rect)
        # Draw the tank head (as a green cell)
        head_rect = pygame.Rect(self.x*CELL_SIZE, self.y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, GREEN, head_rect)

class Obstacle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def draw(self, surface):
        rect = pygame.Rect(self.x*CELL_SIZE, self.y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, GRAY, rect)

class Resource:
    def __init__(self, x, y, type):
        self.x = x
        self.y = y
        self.type = type
    def draw(self, surface):
        rect = pygame.Rect(self.x*CELL_SIZE, self.y*CELL_SIZE, CELL_SIZE, CELL_SIZE)
        # Fuel is blue; ammo is red
        color = BLUE if self.type == 'fuel' else RED
        pygame.draw.rect(surface, color, rect)

class Bullet:
    def __init__(self, x, y, direction, owner):
        self.x = x
        self.y = y
        self.direction = direction
        self.owner = owner
        self.speed = 1
    def update(self):
        self.x += self.direction[0] * self.speed
        self.y += self.direction[1] * self.speed
    def draw(self, surface):
        rect = pygame.Rect(self.x*CELL_SIZE, self.y*CELL_SIZE, CELL_SIZE//2, CELL_SIZE//2)
        pygame.draw.rect(surface, WHITE, rect)

class EnemyTank:
    def __init__(self, x, y, target, bullet_list):
        self.pos = pygame.Vector2(x, y)
        self.angle = 0
        self.speed = 0.5
        self.rotation_speed = 5
        self.target = target
        self.bullet_list = bullet_list
        self.shoot_cooldown = 0
        self.size = CELL_SIZE
    def update(self):
        # Calculate angle to the target (player tank)
        target_vector = pygame.Vector2(self.target.x, self.target.y) - self.pos
        target_angle = math.degrees(math.atan2(target_vector.y, target_vector.x))
        # Adjust angle smoothly
        angle_diff = (target_angle - self.angle + 360) % 360
        if angle_diff > 180:
            angle_diff -= 360
        if abs(angle_diff) > self.rotation_speed:
            self.angle += self.rotation_speed if angle_diff > 0 else -self.rotation_speed
            self.angle %= 360
        else:
            self.angle = target_angle
        # Move in the current facing direction
        rad = math.radians(self.angle)
        direction = pygame.Vector2(math.cos(rad), math.sin(rad))
        self.pos += direction * self.speed
        # Shooting mechanism with cooldown
        if self.shoot_cooldown <= 0:
            self.shoot()
            self.shoot_cooldown = FPS * 2  # reset cooldown (2 seconds)
        else:
            self.shoot_cooldown -= 1
    def shoot(self):
        bx = int(round(self.pos.x + math.cos(math.radians(self.angle))))
        by = int(round(self.pos.y + math.sin(math.radians(self.angle))))
        bullet_direction = (int(round(math.cos(math.radians(self.angle)))),
                            int(round(math.sin(math.radians(self.angle)))))
        self.bullet_list.append(Bullet(bx, by, bullet_direction, 'enemy'))
    def draw(self, surface):
        rect = pygame.Rect(int(self.pos.x)*CELL_SIZE, int(self.pos.y)*CELL_SIZE, self.size, self.size)
        pygame.draw.rect(surface, RED, rect)

class HUD:
    def __init__(self, surface):
        self.surface = surface
        self.font = pygame.font.SysFont('Arial', 24)
    def draw(self, health, fuel, score):
        hud_rect = pygame.Rect(0, 0, SCREEN_WIDTH, 30)
        pygame.draw.rect(self.surface, HUD_BG_COLOR, hud_rect)
        health_text = self.font.render(f"Health: {health}", True, RED)
        fuel_text = self.font.render(f"Fuel: {int(fuel)}", True, YELLOW)
        score_text = self.font.render(f"Score: {int(score)}", True, GREEN)
        self.surface.blit(health_text, (20, 5))
        self.surface.blit(fuel_text, (250, 5))
        self.surface.blit(score_text, (480, 5))

#----------------------------------
# Game Engine
#----------------------------------
class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Grid Battlefield")
        self.clock = pygame.time.Clock()
        self.hud = HUD(self.screen)
        self.init_game_elements()
    def init_game_elements(self):
        self.player = PlayerTank(GRID_WIDTH//2, GRID_HEIGHT//2)
        self.obstacles = []
        self.resources = []
        self.bullets = []
        self.enemies = []
        self.create_obstacles(30)
        self.create_resources(10)
        self.create_enemies(3)
    def create_obstacles(self, count):
        for _ in range(count):
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            self.obstacles.append(Obstacle(x, y))
    def create_resources(self, count):
        for _ in range(count):
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            resource_type = random.choice(['fuel', 'ammo'])
            self.resources.append(Resource(x, y, resource_type))
    def create_enemies(self, count):
        for _ in range(count):
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            self.enemies.append(EnemyTank(x, y, self.player, self.bullets))
    def grid_collision(self, x, y):
        if x < 0 or x >= GRID_WIDTH or y < 0 or y >= GRID_HEIGHT:
            return True
        for obs in self.obstacles:
            if obs.x == x and obs.y == y:
                return True
        return False
    def resource_at(self, x, y):
        for r in self.resources:
            if r.x == x and r.y == y:
                return r
        return None
    def bullet_collision(self, bullet):
        if bullet.x < 0 or bullet.x >= GRID_WIDTH or bullet.y < 0 or bullet.y >= GRID_HEIGHT:
            return True
        for obs in self.obstacles:
            if obs.x == bullet.x and obs.y == bullet.y:
                return True
        return False
    def handle_collisions(self):
        # Collision between the player and obstacles or self-collision
        if self.grid_collision(self.player.x, self.player.y):
            self.player.health -= 10
        if self.player.check_self_collision():
            self.player.health -= 10
        # Resource collection
        res = self.resource_at(self.player.x, self.player.y)
        if res:
            if res.type == 'fuel':
                self.player.fuel = min(100, self.player.fuel + 20)
            elif res.type == 'ammo':
                self.player.score += 10
            self.resources.remove(res)
        # Bullet collisions: enemy bullets hit the player; player bullets hit enemy tanks.
        for bullet in self.bullets[:]:
            if bullet.owner == 'enemy':
                if bullet.x == self.player.x and bullet.y == self.player.y:
                    self.player.health -= 20
                    self.bullets.remove(bullet)
            elif bullet.owner == 'player':
                for enemy in self.enemies[:]:
                    ex = int(round(enemy.pos.x))
                    ey = int(round(enemy.pos.y))
                    if bullet.x == ex and bullet.y == ey:
                        self.enemies.remove(enemy)
                        self.player.score += 50
                        if bullet in self.bullets:
                            self.bullets.remove(bullet)
    def update_bullets(self):
        for bullet in self.bullets[:]:
            bullet.update()
            if self.bullet_collision(bullet):
                if bullet in self.bullets:
                    self.bullets.remove(bullet)
    def shoot_bullet(self, owner):
        if owner == 'player':
            bx = self.player.x + self.player.direction[0]
            by = self.player.y + self.player.direction[1]
            self.bullets.append(Bullet(bx, by, self.player.direction, 'player'))
    def draw_grid(self):
        for x in range(0, SCREEN_WIDTH, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, CELL_SIZE):
            pygame.draw.line(self.screen, GRAY, (0, y), (SCREEN_WIDTH, y))
    def draw_hud(self):
        self.hud.draw(self.player.health, self.player.fuel, self.player.score)
    def run(self):
        while True:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.player.steer("LEFT")
                    elif event.key == pygame.K_RIGHT:
                        self.player.steer("RIGHT")
                    elif event.key == pygame.K_SPACE:
                        self.shoot_bullet('player')
            # Update game elements
            self.player.update()
            for enemy in self.enemies:
                enemy.update()
            self.update_bullets()
            self.handle_collisions()
            # Fuel consumption and score increment over time
            self.player.fuel -= 0.5
            if self.player.fuel <= 0:
                self.player.health -= 5
            self.player.score += 0.1
            # Render drawing: grid, obstacles, resources, player, enemy, bullets and HUD overlay.
            self.screen.fill(BLACK)
            self.draw_grid()
            for obs in self.obstacles:
                obs.draw(self.screen)
            for res in self.resources:
                res.draw(self.screen)
            self.player.draw(self.screen)
            for enemy in self.enemies:
                enemy.draw(self.screen)
            for bullet in self.bullets:
                bullet.draw(self.screen)
            self.draw_hud()
            pygame.display.update()

#----------------------------------
# Integration Testing (Optional)
#----------------------------------
def run_integration_tests():
    game = Game()
    initial_health = game.player.health
    initial_fuel = game.player.fuel
    initial_score = game.player.score

    # Test obstacle collision reduces health
    obs = Obstacle(game.player.x + game.player.direction[0], game.player.y + game.player.direction[1])
    game.obstacles.append(obs)
    game.player.update()
    game.handle_collisions()
    assert game.player.health < initial_health, "Player health did not decrease on obstacle collision."

    # Test resource collection increases fuel
    res = Resource(game.player.x, game.player.y, 'fuel')
    game.resources.append(res)
    pre_fuel = game.player.fuel
    game.handle_collisions()
    assert game.player.fuel > pre_fuel, "Fuel resource not collected properly."

    # Test enemy bullet collision reduces health
    bullet = Bullet(game.player.x, game.player.y, (0,0), 'enemy')
    game.bullets.append(bullet)
    pre_health = game.player.health
    game.handle_collisions()
    assert game.player.health < pre_health, "Enemy bullet collision did not reduce health."

    # Test enemy shooting produces a new bullet
    enemy = EnemyTank(0, 0, game.player, game.bullets)
    pre_bullet_count = len(game.bullets)
    enemy.shoot()
    assert len(game.bullets) > pre_bullet_count, "Enemy shooting did not add a bullet."

    print("All integration tests passed.")

#----------------------------------
# Main Entrypoint
#----------------------------------
if __name__ == "__main__":

    Game().run()

