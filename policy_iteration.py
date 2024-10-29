import numpy as np
from grid_env import GridEnvironment

# Constants
GRID_SIZE = 100
gamma = 0.9  # Discount factor
theta = 1e-4  # Convergence threshold
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

def policy_evaluation(values, policy, env):
    """ Evaluate the current policy until convergence """
    while True:
        delta = 0
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                state = (x, y)
                
                # Skip goal and obstacle cells
                if state == env.goal or env.grid[x, y] == 1:
                    continue
                
                v = values[x, y]  # Store old value to calculate change
                action = actions[policy[x, y]]
                next_state = (x + action[0], y + action[1])
                
                # Check if the next state is valid
                if env.is_valid(next_state):
                    reward = -1 if next_state != env.goal else 100
                    values[x, y] = reward + gamma * values[next_state]
                else:
                    values[x, y] = -1 + gamma * values[x, y]  # Stay in place if invalid move
                
                delta = max(delta, abs(v - values[x, y]))
        
        # If the change is small enough, break
        if delta < theta:
            break

def policy_improvement(values, policy, env):
    """ Improve the policy by selecting actions that maximize expected reward """
    policy_stable = True
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            state = (x, y)
            
            # Skip goal and obstacle cells
            if state == env.goal or env.grid[x, y] == 1:
                continue
            
            old_action = policy[x, y]
            action_values = []
            
            # Evaluate each possible action
            for i, action in enumerate(actions):
                next_state = (x + action[0], y + action[1])
                
                # Calculate expected value based on action
                if env.is_valid(next_state):
                    reward = -1 if next_state != env.goal else 100
                    action_value = reward + gamma * values[next_state]
                else:
                    action_value = -1 + gamma * values[x, y]  # Penalty for invalid moves

                action_values.append(action_value)
            
            # Select the best action for the policy
            best_action = np.argmax(action_values)
            policy[x, y] = best_action
            
            # Check if the policy has changed
            if old_action != best_action:
                policy_stable = False

    return policy_stable

def policy_iteration(env):
    """ Perform Policy Iteration to find the optimal policy """
    values = np.zeros((GRID_SIZE, GRID_SIZE))
    policy = np.random.choice(len(actions), size=(GRID_SIZE, GRID_SIZE))
    
    while True:
        policy_evaluation(values, policy, env)
        if policy_improvement(values, policy, env):
            break
    
    return values, policy
