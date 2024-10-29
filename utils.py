import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def plot_grid(env):
    grid = env.grid
    plt.figure(figsize=(10, 10))

    # Create a custom colormap
    color_map = ListedColormap(['white', 'black', 'green', 'red'])  # Open, Obstacle, Start, Goal
    grid_color = np.zeros(grid.shape)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if (i, j) == env.start:
                grid_color[i, j] = 2  # Start
            elif (i, j) == env.goal:
                grid_color[i, j] = 3  # Goal
            elif grid[i, j] == 1:
                grid_color[i, j] = 1  # Obstacle
            else:
                grid_color[i, j] = 0  # Open cell

    plt.imshow(grid_color, cmap=color_map, interpolation='none')
    plt.title("Grid with Obstacles (Black), Start (Green), and Goal (Red)")
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_values(values):
    plt.imshow(values, cmap="hot")
    plt.colorbar(label="State Value")
    plt.show()
