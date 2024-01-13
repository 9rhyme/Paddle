import gym
from gym import spaces
import numpy as np

class PaddleGameEnv(gym.Env):
    def __init__(self, width=10, height=5, paddle_width=2, ball_radius=1):
        super(PaddleGameEnv, self).__init__()

        self.width = width
        self.height = height
        self.paddle_width = paddle_width
        self.ball_radius = ball_radius

        self.paddle_position = self.width // 2
        self.ball_position = [np.random.randint(self.width), self.height - 1]
        self.ball_velocity = [np.random.choice([-1, 1]), -1]  # Random initial direction

        self.action_space = spaces.Discrete(3)  # 0: Move left, 1: Stay, 2: Move right
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.width + 2, self.height + 2), dtype=np.float32)

    def reset(self):
        self.paddle_position = self.width // 2
        self.ball_position = [np.random.randint(self.width), self.height - 1]
        self.ball_velocity = [np.random.choice([-1, 1]), -1]
        return self._get_observation()

    def step(self, action):
        # Update paddle position based on action
        if action == 0:
            self.paddle_position = max(0, self.paddle_position - 1)
        elif action == 2:
            self.paddle_position = min(self.width - self.paddle_width, self.paddle_position + 1)

        # Update ball position
        self.ball_position[0] += self.ball_velocity[0]
        self.ball_position[1] += self.ball_velocity[1]

        # Reflect ball on walls
        if self.ball_position[0] == 0 or self.ball_position[0] == self.width - 1:
            self.ball_velocity[0] *= -1

        # Reflect ball on ceiling
        if self.ball_position[1] == 0:
            self.ball_velocity[1] *= -1

        # Check if the ball hits the paddle
        if (
            self.paddle_position <= self.ball_position[0] < self.paddle_position + self.paddle_width
            and self.ball_position[1] == self.height - 1
        ):
            self.ball_velocity[1] *= -1

        # Check if the ball is out of bounds
        done = self.ball_position[1] < 0

        return self._get_observation(), 0 if not done else -1, done, {}

    def _get_observation(self):
        observation = np.zeros((self.width + 2, self.height + 2), dtype=np.float32)
        observation[self.paddle_position : self.paddle_position + self.paddle_width, -1] = 1.0
        observation[self.ball_position[0], self.ball_position[1]] = 1.0
        return observation

    def render(self):
        # Simple text-based rendering
        for y in range(self.height + 2):
            row = ""
            for x in range(self.width + 2):
                if x == 0 or x == self.width + 1:
                    row += "|"
                elif y == self.height + 1:
                    row += "-"
                elif y == self.height and self.paddle_position <= x < self.paddle_position + self.paddle_width:
                    row += "="
                elif y == self.ball_position[1] and x == self.ball_position[0]:
                    row += "O"
                else:
                    row += " "
            print(row)





import time

env = PaddleGameEnv()

# Reset the environment to get the initial state
obs = env.reset()

# Run the environment for a few steps
for _ in range(50):
    env.render()
    action = env.action_space.sample()  # Replace with your own action logic
    obs, reward, done, _ = env.step(action)
    time.sleep(0.1)  # Adjust the sleep time for visualization
    if done:
        obs = env.reset()

env.close()







