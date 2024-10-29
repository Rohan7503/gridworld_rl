import numpy as np
import time
from grid_env import GridEnvironment
from policy_iteration import policy_iteration
from q_learning import q_learning
from utils import plot_grid, plot_values

def main():
    # Initialize the environment
    env = GridEnvironment()
    print("Grid Environment with Obstacles and Goal:")
    plot_grid(env)

    # Run Policy Iteration
    print("\nRunning Policy Iteration...")
    start_time = time.time()
    values_pi, policy_pi = policy_iteration(env)
    pi_time = time.time() - start_time
    print("Policy Iteration - Optimal State Values:")
    plot_values(values_pi)
    print("Policy Iteration - Optimal Policy:")
    plot_values(policy_pi)
    print(f"Policy Iteration Time: {pi_time:.4f} seconds")

    # Run Q-learning
    print("\nRunning Q-learning...")
    start_time = time.time()
    Q = q_learning(env, episodes=10000)
    q_time = time.time() - start_time
    optimal_policy_q = np.argmax(Q, axis=2)
    print("Q-learning - Optimal Policy:")
    plot_values(optimal_policy_q)
    print(f"Q-learning Time: {q_time:.4f} seconds")

if __name__ == "__main__":
    main()
