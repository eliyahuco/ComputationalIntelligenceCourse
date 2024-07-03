# Daniel Leiberman - 208206946, Eliyahu Cohen - XXX

# In this task, we planned a pressure vessel with parameters such that would minimize its cost while maintaining constraints
# We did so using PSO and another nature inspired algorithm of our choice
# The other algorithm we choose is Bat Algorithm for the following reasons:
    # Offers good balance between exploration and exploitation
    # Has fewer parameters to tune compared to other algorithms
    # Known for good performance in continuous optimization problems
# We also considered the best set of parameters for each algorithm using a grid search

import numpy as np
import random
import matplotlib.pyplot as plt
from itertools import product

# Cost function
def cost_func(x):
    Ts, Th, R, L = x
    return 0.6224 * Ts * R * L + 1.7787 * Th * R ** 2 + 3.1661 * Ts ** 2 * L + 19.84 * Ts ** 2 * R


# Constraints definition & check
def constraints(x):
    Ts, Th, R, L = x
    g1 = -Ts + 0.0193 * R  # Cylinder thickness constraint
    g2 = -Th + 0.00954 * R  # Hemisphere thickness constraint
    g3 = -np.pi * R ** 2 * L - (4 / 3) * np.pi * R ** 3 + 1_296_000  # Volume constraint
    g4 = L - 240  # Length constraint
    return np.all([g1 <= 0, g2 <= 0, g3 <= 0, g4 <= 0])


# PSO algorithm
def pso(num_particles, num_iterations, w, c1, c2):
    particles = []
    used_positions = set()  # We assured that there is no particle/set of parameters repeating itself
                            # No two (or more) particles occupying the same space at the same time
    while len(particles) < num_particles:
        position = (random.uniform(0, 99), random.uniform(0, 99),
                    random.uniform(10, 200), random.uniform(10, 200))
        if position not in used_positions:
            used_positions.add(position)
            particles.append({'position': np.array(position), 'velocity': np.zeros(4), 'best_position': np.array(position), 'best_score': float('inf')})
    global_best_position = None
    global_best_score = float('inf')
    cost_history = []

    # Initial check to set the global best position and score
    for particle in particles:
        if constraints(particle['position']):
            score = cost_func(particle['position'])
            if score < global_best_score:
                global_best_score = score
                global_best_position = particle['position'].copy()

    for _ in range(num_iterations):
        for particle in particles:
            if constraints(particle['position']):
                score = cost_func(particle['position'])

                if score < particle['best_score']:
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = particle['position'].copy()

        if global_best_position is None:
            # If no valid global best position was found, skip updating velocities and particles
            continue

        for particle in particles:
            r1, r2 = np.random.rand(2)
            particle['velocity'] = (w * particle['velocity'] +
                                    c1 * r1 * (particle['best_position'] - particle['position']) +
                                    c2 * r2 * (global_best_position - particle['position']))
            particle['position'] += particle['velocity']
            particle['position'] = np.clip(particle['position'], [0, 0, 10, 10], [99, 99, 200, 200])

        cost_history.append(global_best_score)

    return global_best_position, global_best_score, cost_history


# Bat Algorithm
def bat_algorithm(num_bats, num_iterations, fmin, fmax, A, r):
    bats = []
    used_positions = set()

    while len(bats) < num_bats:
        position = (random.uniform(0, 99), random.uniform(0, 99), random.uniform(10, 200), random.uniform(10, 200))
        if position not in used_positions:
            used_positions.add(position)
            bats.append({'position': np.array(position), 'velocity': np.zeros(4), 'frequency': np.zeros(4), 'best_position': np.array(position),'best_score': float('inf')})
    global_best_position = None
    global_best_score = float('inf')
    cost_history = []

    # Initial check to set the global best position and score
    for bat in bats:
        if constraints(bat['position']):
            score = cost_func(bat['position'])
            if score < global_best_score:
                global_best_score = score
                global_best_position = bat['position'].copy()

    for _ in range(num_iterations):
        for bat in bats:
            beta = np.random.rand()
            bat['frequency'] = fmin + (fmax - fmin) * beta
            if global_best_position is not None:
                bat['velocity'] = (bat['velocity'] +
                                   (bat['position'] - global_best_position) * bat['frequency'])
                new_position = bat['position'] + bat['velocity']
                if np.random.rand() > r:
                    new_position = global_best_position + 0.001 * np.random.randn(4)

                new_position = np.clip(new_position, [0, 0, 10, 10], [99, 99, 200, 200])

                if constraints(new_position):
                    new_score = cost_func(new_position)
                    if new_score < bat['best_score'] and np.random.rand() < A:
                        bat['best_score'] = new_score
                        bat['best_position'] = new_position

                    if new_score < global_best_score:
                        global_best_score = new_score
                        global_best_position = new_position.copy()

                bat['position'] = new_position

        cost_history.append(global_best_score)

    return global_best_position, global_best_score, cost_history


# Grid search function
def grid_search(algorithm, param_grid, num_particles, num_iterations):
    best_params = None
    best_score = float('inf')
    for params in product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        if algorithm._name_ == 'pso':
            _, score, _ = algorithm(num_particles, num_iterations, **param_dict)
        else:
            _, score, _ = algorithm(num_particles, num_iterations, **param_dict)
        if score < best_score:
            best_score = score
            best_params = param_dict
    return best_params, best_score



num_particles = 50
num_iterations = 1000

# Grid search for PSO
pso_param_grid = {'w': [0.5, 0.7, 0.9], 'c1': [1.0, 1.5, 2.0], 'c2': [1.0, 1.5, 2.0]}
best_pso_params, best_pso_score = grid_search(pso, pso_param_grid, num_particles, num_iterations)
print(f"Best PSO parameters: {best_pso_params}")
print(f"Best PSO score: {best_pso_score}")

# Run PSO with best parameters and print optimal values
best_position, best_score, pso_history = pso(num_particles, num_iterations, **best_pso_params)
print(f"Optimal PSO values: T_s={best_position[0]}, T_h={best_position[1]}, R={best_position[2]}, L={best_position[3]}")

# Grid search for Bat Algorithm
ba_param_grid = {'fmin': [0, 0.5], 'fmax': [1, 2], 'A': [0.5, 0.7, 0.9], 'r': [0.1, 0.3, 0.5]}
best_ba_params, best_ba_score = grid_search(bat_algorithm, ba_param_grid, num_particles, num_iterations)
print(f"Best Bat Algorithm parameters: {best_ba_params}")
print(f"Best Bat Algorithm score: {best_ba_score}")

# Run Bat Algorithm with best parameters and print optimal values
best_position, best_score, ba_history = bat_algorithm(num_particles, num_iterations, **best_ba_params)
print(f"Optimal Bat Algorithm values: T_s={best_position[0]}, T_h={best_position[1]}, R={best_position[2]}, L={best_position[3]}")

plt.figure(figsize=(10, 6))
plt.plot(pso_history, label='PSO')
plt.plot(ba_history, label='Bat Algorithm')
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs Iterations')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# Main execution
if __name__ == "_main_":
    num_particles = 50
    num_iterations = 1000

    # Grid search for PSO
    pso_param_grid = {'w': [0.5, 0.7, 0.9], 'c1': [1.0, 1.5, 2.0], 'c2': [1.0, 1.5, 2.0]}
    best_pso_params, best_pso_score = grid_search(pso, pso_param_grid, num_particles, num_iterations)
    print(f"\nBest PSO parameters: {best_pso_params}")
    print(f"Best PSO score: {best_pso_score}")

    # Run PSO with best parameters and print optimal values
    best_position, best_score, pso_history = pso(num_particles, num_iterations, **best_pso_params)
    print(f"\nOptimal PSO values: T_s={best_position[0]}, T_h={best_position[1]}, R={best_position[2]}, L={best_position[3]}")

    # Grid search for Bat Algorithm
    ba_param_grid = {'fmin': [0, 0.5], 'fmax': [1, 2], 'A': [0.5, 0.7, 0.9], 'r': [0.1, 0.3, 0.5]}
    best_ba_params, best_ba_score = grid_search(bat_algorithm, ba_param_grid, num_particles, num_iterations)
    print(f"\nBest Bat Algorithm parameters: {best_ba_params}")
    print(f"Best Bat Algorithm score: {best_ba_score}")

    # Run Bat Algorithm with best parameters and print optimal values
    best_position, best_score, ba_history = bat_algorithm(num_particles, num_iterations, **best_ba_params)
    print(f"\nOptimal Bat Algorithm values: T_s={best_position[0]}, T_h={best_position[1]}, R={best_position[2]}, L={best_position[3]}")

    plt.figure(figsize=(10, 6))
    plt.plot(pso_history, label='PSO')
    plt.plot(ba_history, label='Bat Algorithm')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost vs Iterations')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.show()