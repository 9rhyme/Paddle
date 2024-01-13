import pygame
import sys
import numpy as np
from time import sleep

# Constants for the screen size
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

# Paddle Constants
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20

# Ball Constants
BALL_DIAMETER = 20

# Block Constants
BLOCK_WIDTH = 35
BLOCK_HEIGHT = 20
NUM_BLOCKS_PER_ROW = 12
NUM_ROWS = 3

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

class Paddle():
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Set up some properties
        self.done = False
        self.reward = 0
        self.hit, self.miss = 0, 0

        # Set up the display
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

        # Set up the paddle
        self.paddle = pygame.Rect(SCREEN_WIDTH / 2, SCREEN_HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)

        # Set up the ball
        self.ball = pygame.Rect(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, BALL_DIAMETER, BALL_DIAMETER)
        self.ball_dx = 3
        self.ball_dy = -3

        # Set up the blocks
        self.blocks = []
        for row in range(NUM_ROWS):
            y_position = row * (BLOCK_HEIGHT + 5) + 50  # Adjust the starting y-position for each row
            row_blocks = [pygame.Rect(x * (BLOCK_WIDTH + 5) + (SCREEN_WIDTH - (NUM_BLOCKS_PER_ROW * (BLOCK_WIDTH + 5))) / 2,
                                      y_position, BLOCK_WIDTH, BLOCK_HEIGHT) for x in range(NUM_BLOCKS_PER_ROW)]
            self.blocks.extend(row_blocks)

        # Set up the font for the score display
        self.font = pygame.font.Font(None, 36)

    def paddle_left(self):
        if self.paddle.left > 0:
            self.paddle.left -= 20

    def paddle_right(self):
        if self.paddle.right < SCREEN_WIDTH:
            self.paddle.right += 20

    def run_frame(self):

        # sleep(0.017) # To slow down the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.paddle_left()
        if keys[pygame.K_RIGHT]:
            self.paddle_right()

        # Update ball position
        self.ball.left += self.ball_dx
        self.ball.top += self.ball_dy

        # Ball and wall collision
        if self.ball.left <= 0 or self.ball.right >= SCREEN_WIDTH:
            self.ball_dx *= -1
        if self.ball.top <= 0:
            self.ball_dy *= -1

        # Ball and paddle collision
        if self.ball.colliderect(self.paddle) and self.ball_dy > 0:
            self.ball_dy *= -1
            self.hit += 1
            self.reward += 10

        # Ball and block collision
        for block in self.blocks:
            if self.ball.colliderect(block):
                self.blocks.remove(block)
                self.ball_dy *= -1
                self.reward += 1 + 2 * block.y // (BLOCK_HEIGHT + 5)  # Higher rows give slightly more rewards

        # Ball misses paddle
        if self.ball.top >= SCREEN_HEIGHT:
            self.ball.center = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)
            self.miss += 1
            self.reward -= 10
            self.done = True

        # Draw everything
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, WHITE, self.paddle)
        pygame.draw.circle(self.screen, RED, self.ball.center, BALL_DIAMETER / 2)
        for block in self.blocks:
            if block.y < BLOCK_HEIGHT + 55:  # First row color
                pygame.draw.rect(self.screen, GREEN, block)
            elif block.y < 2 * (BLOCK_HEIGHT + 5) + 55:  # Second row color
                pygame.draw.rect(self.screen, BLUE, block)
            else:  # Third row color
                pygame.draw.rect(self.screen, RED, block)

        # Render the score
        hit_percentage = self.hit / (self.hit + self.miss) * 100 if self.hit + self.miss > 0 else 0
        score_text = self.font.render("Hits: {}   Missed: {}".format(self.hit, self.miss), True, WHITE)
        self.screen.blit(score_text, (20, 20))
        stats_text = self.font.render("Accuracy: {:.2f}%".format(hit_percentage), True, WHITE)
        self.screen.blit(stats_text, (SCREEN_WIDTH - stats_text.get_width() - 20, 20))

        # Flip the display
        pygame.display.flip()

    # ------------------------ AI control ------------------------

    # 0 move left
    # 1 do nothing
    # 2 move right

    def reset(self):
        # place the paddle at the center
        self.paddle.center = (SCREEN_WIDTH / 2, SCREEN_HEIGHT - PADDLE_HEIGHT - 10)

        # place the ball at center vertically but randomly horizontally
        self.ball.center = (np.random.randint(0, SCREEN_WIDTH), SCREEN_HEIGHT / 2)

        # reset the blocks
        self.blocks = []
        for row in range(NUM_ROWS):
            y_position = row * (BLOCK_HEIGHT + 5) + 50
            row_blocks = [pygame.Rect(x * (BLOCK_WIDTH + 5) + (SCREEN_WIDTH - (NUM_BLOCKS_PER_ROW * (BLOCK_WIDTH + 5))) / 2,
                                      y_position, BLOCK_WIDTH, BLOCK_HEIGHT) for x in range(NUM_BLOCKS_PER_ROW)]
            self.blocks.extend(row_blocks)

        return [self.paddle.centerx / 600, self.ball.centerx / 600, self.ball.centery / 600, self.ball_dx,
                self.ball_dy]

    # Step
    def step(self, action):
        self.reward = 0
        self.done = 0
        if action == 0:
            self.paddle_left()
            self.reward -= .1
        if action == 2:
            self.paddle_right()
            self.reward -= .1

        self.run_frame()
        state = [self.paddle.centerx / 600, self.ball.centerx / 600, self.ball.centery / 600, self.ball_dx,
                 self.ball_dy]
        return self.reward, state, self.done




if __name__ == "__main__":

    env = Paddle()
    clock = pygame.time.Clock()

    while not env.done:
        env.run_frame()
        clock.tick(60)  # Limit to 60 frames per second
    pygame.quit()
