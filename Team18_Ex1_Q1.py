"""Authors: Eliyahu cohen, id 304911084
            Daniel liberman, id 208206946

---------------------------------------------------------------------------------
Short Description:

This script is the HW_1 in the course Computational Intelligence
The script is an implementation of a genetic algorithm to solve the knapsack problem.
The knapsack problem is a combinatorial optimization problem that seeks to maximize the profit of items in a knapsack without exceeding the weight limit.
The script uses a genetic algorithm to find the best combination of items that maximizes the profit while keeping the total weight below the weight limit.
The script uses three different selection methods: roulette wheel, tournament, and ranking and mutation.
any generation we will choose only two parents to crossover and create two offspring.
The script plots a convergence graph showing the maximum profit obtained in each generation.
The script prints the best solution found by the genetic algorithm, the expected profit, total weight, and the included instruments.
the script will run with stop condition of 1000 generations or 50 generations do not improve the best solution.

---------------------------------------------------------------------------------
"""
#Labraries in use

import random
import matplotlib.pyplot as plt

# Parameters
population_size = 50
generations = 1000
mutation_rate = 0.002
crossover_rate = 0.9999
weight_limit = 100

# Instruments data: [profit, weight]


# [profit, weight]
instruments = [[1212, 2.91],[1211, 8.19],[612, 5.55],[609, 15.6],
    [1137, 3.70],[1300, 13.5],[585, 14.9],[1225, 7.84],[1303, 17.6],
    [728, 17.3],[211, 6.83],[336, 14.4],[894, 2.11],[1381, 7.25],
    [597, 4.65],[858, 17.0],[854, 7.28],[1156, 5.01],[597, 16.1],
    [1129, 16.7],[850, 3.10],[874, 6.77],[579, 10.7],[1222, 1.25],
    [896, 17.2]
]


W = 100

# You can add your logic here to process the instruments and the total weight W
print(f"Total instruments: {len(instruments)}")
print(f"Total weight limit: {W}")


W = 100
# Initialize population
def initialize_population(size, n_items):
    return [[random.randint(0, 1) for _ in range(n_items)] for _ in range(size)]


# Calculate fitness
def calculate_fitness(individual):
    total_weight = sum(individual[i] * instruments[i][1] for i in range(len(individual)))
    total_profit = sum(individual[i] * instruments[i][0] for i in range(len(individual)))
    if total_weight > weight_limit:
        return 0
    return total_profit


# Selection using roulette wheel
def selection(population, fitnesses):
    max_fitness = sum(fitnesses)
    pick = random.uniform(0, max_fitness)
    current = 0
    for individual, fitness in zip(population, fitnesses):
        current += fitness
        if current > pick:
            return individual


# Tournament selection
def tournament_selection(population, fitnesses, k=3):
    selected = random.sample(list(zip(population, fitnesses)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]


# Ranking selection
def ranking_selection(population, fitnesses):
    sorted_population = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
    rank_probabilities = [1 / (i + 1) for i in range(len(sorted_population))]
    total = sum(rank_probabilities)
    pick = random.uniform(0, total)
    current = 0
    for rank, individual in zip(rank_probabilities, sorted_population):
        current += rank
        if current > pick:
            return individual


# Crossover
def crossover(parent1, parent2):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]
    return parent1, parent2


# Mutation
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]
    return individual


# Moving average for smoothing
def moving_average(data, window_size):
    return [sum(data[i:i + window_size]) / window_size for i in range(len(data) - window_size + 1)]


# Genetic Algorithm
def genetic_algorithm(selection_method):
    population = initialize_population(population_size, len(instruments))
    max_profits = []
    mutation_rate = 1
    crossover_rate = 0.001
    for generation in range(generations):
        rand = random.random()
        if generation % 100 == 0 and generation > 0:
            print("Generation", generation, "Max profit:", max(max_profits) if max_profits else 0,"total weight:", sum(max(population, key=calculate_fitness)[i] * instruments[i][1] for i in range(len(instruments))))

            mutation_rate = mutation_rate * 0.9
            crossover_rate = crossover_rate * 1.1
        fitnesses = [calculate_fitness(individual) for individual in population]
        max_profits.append(max(fitnesses))
        if generation > 50 and max_profits[-1] == max_profits[-50] and max_profits[-1] >= max(max_profits):
            for i in range(generation, generations-1):
                max_profits.append(max_profits[-1])
            break
        new_population = []

        for _ in range(population_size // 2):

            if selection_method == 'roulette':
                parent1 = selection(population, fitnesses)
                parent2 = selection(population, fitnesses)

            elif selection_method == 'tournament':
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
            elif selection_method == 'ranking':
                parent1 = ranking_selection(population, fitnesses)
                parent2 = ranking_selection(population, fitnesses)
            elif selection_method == 'mixed':

                if rand < 0.33:
                    parent1 = selection(population, fitnesses)
                    parent2 = selection(population, fitnesses)
                elif rand < 0.66:
                    parent1 = tournament_selection(population, fitnesses)
                    parent2 = tournament_selection(population, fitnesses)
                else:
                    parent1 = ranking_selection(population, fitnesses)
                    parent2 = ranking_selection(population, fitnesses)

            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])

        population = new_population

        plt.clf()
        plt.xlabel('Generation')
        plt.ylabel('Max Profit')
        plt.title('Convergence Graph')
        plt.plot(range(generation + 1), max_profits, label='Max Profit')
        plt.pause(0.1)

    best_individual = max(population, key=calculate_fitness)
    best_fitness = calculate_fitness(best_individual)
    total_weight = sum(best_individual[i] * instruments[i][1] for i in range(len(best_individual)))

    print("GA found best solution after", generation, "iterations:")
    print("Expected profit =", best_fitness)
    print("Total weight =", total_weight)
    print("Included instruments are", [i for i in range(len(best_individual)) if best_individual[i] == 1])

    # Plot convergence graph with grid and smoothed line
    plt.plot(range(generations), max_profits, label='Max Profit')

    # Smooth the curve using moving average
    window_size = 10
    smoothed_profits = moving_average(max_profits, window_size)
    plt.plot(range(window_size - 1, generations), smoothed_profits, label='Smoothed Max Profit', color='red')


    plt.xlabel('Generation')
    plt.ylabel('Max Profit')
    plt.title('Convergence Graph')
    plt.grid(True)
    plt.legend()
    plt.show()



Run the genetic algorithm with different selection methods
print("Running GA with Roulette Wheel Selection\n")
genetic_algorithm('roulette')


print("\nRunning GA with Tournament Selection\n")
genetic_algorithm('tournament')

print("\nRunning GA with Ranking Selection\n")
genetic_algorithm('ranking')

print("\nRunning GA with Random Selection\n")
genetic_algorithm('mixed')



