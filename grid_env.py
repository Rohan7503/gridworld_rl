import numpy as np
import random

GRID_SIZE = 100
OBSTACLE_DENSITY = 0.2

class GridEnvironment:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self.start = (0, 0)
        self.goal = (GRID_SIZE - 1, GRID_SIZE - 1)
        self.grid[self.start] = 2
        self.grid[self.goal] = 3
        self._place_obstacles()

    def _place_obstacles(self):
        num_obstacles = int(GRID_SIZE * GRID_SIZE * OBSTACLE_DENSITY)
        for _ in range(num_obstacles):
            x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
            if (x, y) not in [self.start, self.goal]:
                self.grid[x, y] = 1

    def is_valid(self, state):
        x, y = state
        return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and self.grid[x, y] != 1
