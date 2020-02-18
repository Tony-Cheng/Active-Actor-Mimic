import numpy as np
import random

UP = 0
LEFT = 1
RIGHT = 2
DOWN = 3

WHITE = 0
GREEN = 1
BLUE = 2
RED=3

NORMAL = 0
GOAL = 1
LAVA = 2
AGENT = 3

class KGW(object):
    def __init__(self, max_steps=50):
        self.grid = np.zeros((9, 9))
        self.max_steps = max_steps

    def reset(self):
        self.grid_color = np.zeros((9, 9))
        self.grid_object = np.zeros((9, 9))
        self.agent_pos = (int(random.random() * 9), int(random.random() * 9))
        self.build_goals_and_lava()
        self.num_steps = 0

    def _random_empty_position(self):
        position = (int(random.random() * 9), int(random.random() * 9))
        while (not self.grid_object[position] == 0) or position == self.agent_pos:
            position = (int(random.random() * 9), int(random.random() * 9))
        return position

    def build_goals_and_lava(self):
        position = self._random_empty_position()
        self.grid_object[position] = GOAL
        self.grid_color[position] = RED
        position = self._random_empty_position()
        self.grid_object[position] = GOAL
        self.grid_color[position] = GREEN
        position = self._random_empty_position()
        self.grid_object[position] = GOAL
        self.grid_color[position] = BLUE
        position = self._random_empty_position()
        self.grid_object[position] = LAVA
        self.grid_color[position] = RED
        position = self._random_empty_position()
        self.grid_object[position] = LAVA
        self.grid_color[position] = GREEN
        position = self._random_empty_position()
        self.grid_object[position] = LAVA
        self.grid_color[position] = BLUE
    
    def step(self, action):
        reward = 0
        done = False
        if action == UP:
            if self.agent_pos[0] > 0:
                self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        if action == LEFT:
            if self.agent_pos[1] > 0:
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        if action == RIGHT:
            if self.agent_pos[1] < 8:
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        if action == DOWN:
            if self.agent_pos[0] < 8:
                self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        if not self.grid_color[self.agent_pos] == 0:
            reward += 1
        if self.grid_object[self.agent_pos] == LAVA or self.num_steps == self.max_steps:
            done = True
        if done:
            obs = None
        else:
            obs = self.build_obs()
        return obs, reward, done, None

    def build_obs(self):
        obs = np.zeros((9, 9, 8))
        obs[:, :, 0] = (self.grid_object == NORMAL) + 0 
        obs[:, :, 1] = (self.grid_object == GOAL) + 0 
        obs[:, :, 2] = (self.grid_object == LAVA) + 0 
        obs[:, :, 3] = (self.grid_color == WHITE) + 0
        obs[:, :, 4] = (self.grid_color == RED) + 0 
        obs[:, :, 5] = (self.grid_color == GREEN) + 0 
        obs[:, :, 6] = (self.grid_color == BLUE) + 0
        obs[self.agent_pos[0], self.agent_pos[1], 7] = 1 
        return obs



         
