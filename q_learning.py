import numpy as np
import random
from grid_env import GridEnvironment

# Constants
GRID_SIZE = 100
gamma = 0.9  # Discount factor
alpha = 0.1  # Learning rate
epsilon = 0.05  # Exploration rate
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

def choose_action(state, Q, env):
    """ Epsilon-greedy action selection """
    if random.uniform(0, 1) < epsilon:
        # Exploration: choose a random action
        return random.choice(range(len(actions)))
    else:
        # Exploitation: choose the best action based on Q-values
        x, y = state
        return np.argmax(Q[x, y])

def take_step(state, action, env):
    """ Take a step in the environment and return the next state and reward """
    x, y = state
    dx, dy = actions[action]
    next_state = (x + dx, y + dy)
    
    # Check if the move is valid
    if env.is_valid(next_state):
        reward = -1 if next_state != env.goal else 100  # Negative reward unless reaching goal
    else:
        next_state = state  # If invalid move, stay in the same place
        reward = -1  # Penalty for hitting an obstacle or boundary
    
    return next_state, reward

def q_learning(env, episodes=1000):
    """ Q-learning algorithm for the grid environment """
    # Initialize Q-table with zeros
    Q = np.zeros((GRID_SIZE, GRID_SIZE, len(actions)))
    
    for episode in range(episodes):
        state = env.start  # Start each episode at the starting point

        while state != env.goal:
            x, y = state
            
            # Choose an action using epsilon-greedy policy
            action = choose_action(state, Q, env)
            
            # Take the action and observe the outcome
            next_state, reward = take_step(state, action, env)
            next_x, next_y = next_state
            
            # Update the Q-value for the current state-action pair
            best_next_action = np.argmax(Q[next_x, next_y])
            Q[x, y, action] = Q[x, y, action] + alpha * (
                reward + gamma * Q[next_x, next_y, best_next_action] - Q[x, y, action]
            )
            
            # Move to the next state
            state = next_state

    return Q
