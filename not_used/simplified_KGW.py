import numpy as np
import random
from enum import Enum


class Action():
    UP = 0
    LEFT = 1
    RIGHT = 2
    DOWN = 3


class Color():
    WHITE = 0
    GREEN = 1
    BLUE = 2
    RED = 3

def color_list():
    return [Color.WHITE, Color.GREEN, Color.BLUE, Color.RED]


class Tile():
    NORMAL = 0
    GOAL = 1
    LAVA = 2
    AGENT = 3


class Objective():
    GOAL_RED = 0


class KGW(object):
    def __init__(self, max_steps=50):
        self.grid = np.zeros((9, 9))
        self.max_steps = max_steps
        self.n_actions = 4

    def reset(self, objective=None):
        self.grid_color = np.zeros((9, 9))
        self.grid_object = np.zeros((9, 9))
        self.num_steps = 0
        self.objective = objective
        self.agent_pos = (int(random.random() * 9), int(random.random() * 9))
        self._build_goals_and_lava()
        self._build_colors()
        return self.build_obs()

    def _random_empty_position(self):
        position = (int(random.random() * 9), int(random.random() * 9))
        while (not self.grid_object[position] == 0) or position == self.agent_pos:
            position = (int(random.random() * 9), int(random.random() * 9))
        return position

    def _build_goals_and_lava(self):
        position = self._random_empty_position()
        self.grid_object[position] = Tile.GOAL
        self.grid_color[position] = Color.RED
        position = self._random_empty_position()
        self.grid_object[position] = Tile.GOAL
        self.grid_color[position] = Color.GREEN
        position = self._random_empty_position()
        self.grid_object[position] = Tile.GOAL
        self.grid_color[position] = Color.BLUE
        position = self._random_empty_position()
        self.grid_object[position] = Tile.LAVA
        self.grid_color[position] = Color.RED
        position = self._random_empty_position()
        self.grid_object[position] = Tile.LAVA
        self.grid_color[position] = Color.GREEN
        position = self._random_empty_position()
        self.grid_object[position] = Tile.LAVA
        self.grid_color[position] = Color.BLUE

    def _build_colors(self):
        colors = color_list()
        for i in range(9):
            for j in range(9):
                while self.grid_color[i, j] == Color.WHITE:
                    self.grid_color[i, j] = colors[int(
                        random.random() * len(colors))]

    def step(self, action):
        self._make_move(action)
        reward = self._calc_reward()
        done = self._check_done()
        self.num_steps += 1
        obs = self.build_obs()
        return obs, reward, done, None

    def build_obs(self):
        obs = np.zeros((8, 9, 9))
        obs[0, :, :] = (self.grid_object == Tile.NORMAL) + 0
        obs[1, :, :] = (self.grid_object == Tile.GOAL) + 0
        obs[2, :, :] = (self.grid_object == Tile.LAVA) + 0
        obs[3, :, :] = (self.grid_color == Color.WHITE) + 0
        obs[4, :, :] = (self.grid_color == Color.RED) + 0
        obs[5, :, :] = (self.grid_color == Color.GREEN) + 0
        obs[6, :, :] = (self.grid_color == Color.BLUE) + 0
        obs[7, self.agent_pos[0], self.agent_pos[1]] = 1
        return obs

    def _make_move(self, action):
        if action == Action.UP:
            if self.agent_pos[0] > 0:
                self.agent_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        if action == Action.LEFT:
            if self.agent_pos[1] > 0:
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        if action == Action.RIGHT:
            if self.agent_pos[1] < 8:
                self.agent_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        if action == Action.DOWN:
            if self.agent_pos[0] < 8:
                self.agent_pos = (self.agent_pos[0] + 1, self.agent_pos[1])

    def _calc_reward(self):
        if self.objective is None:
            if self.grid_object[self.agent_pos] == Tile.GOAL:
                return 1.0
            else:
                return 0.0
    def _check_done(self):
        if self.grid_object[self.agent_pos] == Tile.LAVA or self.num_steps >= self.max_steps:
            return True
        else:
            return False
