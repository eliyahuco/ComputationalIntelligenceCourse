"""Authors: Eliyahu cohen, id 304911084
            Daniel liberman, id 208206946

---------------------------------------------------------------------------------
Short Description:

This script is the HW_1 in the course Computational Intelligence
---------------------------------------------------------------------------------
"""
#Labraries in use
import random
import matplotlib.pyplot as plt

# Parameters
population_size = 50
generations = 1000
mutation_rate = 0.01
crossover_rate = 0.7
weight_limit = 100

# Instruments data: [profit, weight]
instruments = [
    [1212, 2.91], [1211, 8.19], [612, 5.55], [609, 15.6], [1073, 19.1],
    [1300, 13.5], [895, 7.21], [1225, 7.84], [1833, 17.6], [728, 7.13],
    [211, 6.83], [894, 11.1], [1311, 14.5], [597, 4.65], [858, 17.01],
    [854, 7.281], [1352, 11.1], [597, 10.15], [1176, 15.6], [1205, 8.01],
    [1178, 10.4], [874, 7.677], [1222, 1.25], [806, 17.2]
]


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

    for generation in range(generations):
        fitnesses = [calculate_fitness(individual) for individual in population]
        max_profits.append(max(fitnesses))
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
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])
        population = new_population

    best_individual = max(population, key=calculate_fitness)
    best_fitness = calculate_fitness(best_individual)
    total_weight = sum(best_individual[i] * instruments[i][1] for i in range(len(best_individual)))

    print("GA found best solution after", generations, "iterations:")
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


# Run the genetic algorithm with different selection methods
print("Running GA with Roulette Wheel Selection")
genetic_algorithm('roulette')

print("Running GA with Tournament Selection")
genetic_algorithm('tournament')

print("Running GA with Ranking Selection")
genetic_algorithm('ranking')