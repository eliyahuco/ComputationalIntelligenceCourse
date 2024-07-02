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
import numpy as np

# Parameters
population_size = 50
generations = 1000
mutation_rate = 0.5
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



# Initialize population
def initialize_population(size, n_items):
    '''
    This function initializes the population with random binary values.
    :param size: population size
    :param n_items: number of items
    :return: a list of lists with random binary values that represent the population
    '''
    return [[random.randint(0, 1) for _ in range(n_items)] for _ in range(size)]


# Calculate fitness
def calculate_fitness(individual):
    '''
    This function calculates the fitness of an individual by summing the profit of the selected items.
    If the total weight exceeds the weight limit, the fitness is set to 0.
    :param individual: a list of binary values representing the selected items
    :return: the fitness value of the individual
    '''

    total_weight = sum(individual[i] * instruments[i][1] for i in range(len(individual)))
    total_profit = sum(individual[i] * instruments[i][0] for i in range(len(individual)))
    if total_weight > weight_limit:
        return 0
    return total_profit



# calculate fitness for all population
def calculate_fitnesses_fo_all_population(population):
    '''
    This function calculates the fitness of all individuals in the population.
    :param population: a list of lists representing the population
    :return: a list of fitness values corresponding to the population
    '''

    fitnesses = []
    for individual in population:
        fitnesses.append(calculate_fitness(individual))
    return fitnesses

#selection methods: random, proportional, tournament, ranking, elite
def random_selection(population):
    '''
    This function selects an individual from the population using roulette wheel selection.
    The probability of selecting an individual is proportional to its fitness value.
    :param population: a list of lists representing the population
    :return: the selected individual
    '''
    len_population = len(population)
    pick = random.uniform(0,len_population )
    return population[int(pick)]
def proportional_selection(population, fitnesses,pick=0.1):
    '''
    This function selects an individual from the population using roulette wheel selection.
    The probability of selecting an individual is proportional to its fitness value.
    :param population: a list of lists representing the population
    :param fitnesses: a list of fitness values corresponding to the population
    :return: the selected individual
    '''
    total_fitness = sum(fitnesses)
    if total_fitness == 0:
        return random.choice(population)
    proportional_list = [fitness / total_fitness for fitness in fitnesses]
    selected_individual = random.choices(population, weights=proportional_list, k=1)[0]

    return selected_individual



# Tournament selection
def tournament_selection(population, fitnesses, k=5):
    '''
    This function selects an individual from the population using tournament selection.
    The function selects k individuals at random and returns the best individual.
    :param population: a list of lists representing the population
    :param fitnesses: a list of fitness values corresponding to the population
    :param k: the number of individuals to select for the tournament
    :return: the selected individual
    '''

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
def crossover(parent1, parent2, crossover_rate=0.5):
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]


    return parent1, parent2


# Mutation using bit flip
def mutate(individual, mutation_rate=0.5):
    '''
    This function mutates an individual by flipping a random bit with a given probability.
    :param individual: a list of binary values representing the individual
    :param mutation_rate: the probability of flipping a bit
    :return: the mutated individual
    '''

    if random.random() < mutation_rate:
        index = random.randint(0, len(individual) - 1)
        individual[index] = 1 - individual[index]
        return individual
    return individual



# Moving average for smoothing
def moving_average(data, window_size):
    return [sum(data[i:i + window_size]) / window_size for i in range(len(data) - window_size + 1)]



def genetic_algorithm_maximize_the_profit(selection_methods,instruments,population , generations=1000, population_size=50, mutation_rate=0.9, crossover_rate=0.1):
    '''
    This function implements a genetic algorithm to solve the knapsack problem.
    The goal is to maximize the profit of items in the knapsack without exceeding the weight limit.
    The function uses different selection methods: random, proportional, tournament, ranking, and elite.
    :param selection_methods: the selection method to use (random, proportional, tournament, ranking, elite)
    :param instruments: a list of lists representing the items to select from
    :param generations: the number of generations to run the algorithm
    :param population_size: the size of the population
    :param mutation_rate: the probability of mutation
    :param crossover_rate: the probability of crossover
    :param weight_limit: the weight limit of the knapsack
    :return: the best individual found by the genetic algorithm, the expected profit, and the total weight
    '''




    max_profits = []
    total_weight_list = []

    fitnesses = calculate_fitnesses_fo_all_population(population)
    offspring_population = [population[0], population[1], population[2], population[3]]
    for generation in range(generations):
        if len(population) >  population_size*1.25:
            for i in range(len(population) - int(population_size*0.4)):
                population.remove(min(population, key=calculate_fitness))
            fitnesses = calculate_fitnesses_fo_all_population(population)


        if selection_methods == 'random':

            parent1 = random_selection(population)
            parent2 = random_selection(population)
            offspring1, offspring2 = crossover(parent1, parent2)
            population.extend([offspring1, offspring2])
            population.extend([mutate(offspring1), mutate(offspring2)])
            # population.extend([mutate(parent1), mutate(parent2)])
            mutation_rate = mutation_rate * 0.75
            crossover_rate = crossover_rate * 1.1



        elif selection_methods == 'proportional':

            parent1 = proportional_selection(population, fitnesses)
            parent2 = proportional_selection(population, fitnesses)
            offspring1, offspring2 = crossover(parent1, parent2)
            population.extend([offspring1, offspring2])
            population.extend([mutate(offspring1), mutate(offspring2)])

            mutation_rate = mutation_rate * 0.9
            crossover_rate = crossover_rate * 1.1


        elif selection_methods == 'tournament':
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            offspring1, offspring2 = crossover(parent1, parent2)

            if calculate_fitness(offspring1) < calculate_fitness(parent1):
                offspring1 = parent1
            if calculate_fitness(offspring2) < calculate_fitness(parent2):
                offspring2 = parent2
            population.extend([offspring1, offspring2])
            population.extend([mutate(offspring1), mutate(offspring2)])
            mutation_rate = mutation_rate * 0.9
            crossover_rate = crossover_rate * 1.1
            offspring_population.extend([offspring1, offspring2])
        elif selection_methods == 'ranking':
            parent1 = ranking_selection(population, fitnesses)
            parent2 = ranking_selection(population, fitnesses)
            offspring1, offspring2 = crossover(parent1, parent2)

            if calculate_fitness(offspring1) < calculate_fitness(parent1):
                offspring1 = parent1
            if calculate_fitness(offspring2) < calculate_fitness(parent2):
                offspring2 = parent2
            population.extend([offspring1, offspring2])
            population.extend([mutate(offspring1), mutate(offspring2)])

            mutation_rate = mutation_rate * 0.75
            crossover_rate = crossover_rate * 1.25
            offspring_population.extend([offspring1, offspring2])
        elif selection_methods == 'elite':
            for i in range(len(population) - int(population_size*0.5)):
                population.remove(min(population, key=calculate_fitness))

            mutation_rate = mutation_rate*0.9
            crossover_rate = crossover_rate*1.1
            parent1 = max(population, key=calculate_fitness)
            parent2 = max(offspring_population, key=calculate_fitness)
            offspring1, offspring2 = crossover(parent1, parent2)
            if calculate_fitness(offspring1) < calculate_fitness(parent1):
                offspring1 = parent1
            if calculate_fitness(offspring2) < calculate_fitness(parent2):
                offspring2 = parent2
            population.extend([offspring1, offspring2])
            population.extend([mutate(offspring1), mutate(offspring2)])
            offspring_population.extend([offspring1, offspring2])








        elif selection_methods == 'progress_selection':
            if generations < 30:
                parent1 = random_selection(population, fitnesses)
                parent2 = random_selection(population, fitnesses)
            elif generations < 50 and generations >= 40:
                parent1 = max(population, key=calculate_fitness)
                parent2 = max(population, key=calculate_fitness)

            elif generations < 120 and generations >= 60:
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
            else:
                parent1 = ranking_selection(population, fitnesses)
                parent2 = ranking_selection(population, fitnesses)

            offspring1, offspring2 = crossover(parent1, parent2)
            population.extend([parent1, parent2])
            population.extend([offspring1, offspring2])
            population.extend([mutate(offspring1), mutate(offspring2)])
            population.remove(min(population, key=calculate_fitness))
            population.remove(min(population, key=calculate_fitness))
            mutation_rate = mutation_rate * 0.9
            crossover_rate = crossover_rate * 1.1

        population.remove(min(population, key=calculate_fitness))
        population.remove(min(population, key=calculate_fitness))

        fitnesses = calculate_fitnesses_fo_all_population(population)
        max_profits.append(max(fitnesses))
        total_weight_list.append(sum(max(population, key=calculate_fitness)[i] * instruments[i][1] for i in range(len(instruments))))

        if mutation_rate < 0.1:
            mutation_rate = 0.9
        if crossover_rate > 0.9:
            crossover_rate = 0.1
        # if  generation % 100 == 0 and generation > 0:
        #     print("\nGeneration", generation, "profit:", max_profits[-1], "total weight:", sum(max(population, key=calculate_fitness)[i] * instruments[i][1] for i in range(len(instruments))))
        #     print("Mutation rate:", mutation_rate, "Crossover rate:", crossover_rate)
        #     print("Best solution so far:", max(max_profits), 'best_weight' , total_weight_list[max_profits.index(max(max_profits ))])

        if generation > 100 and max_profits[-1] == max_profits[-50]  and max_profits[-1] >= max(max_profits):
            for i in range(generation, generations-1):
                max_profits.append(max_profits[-1])
            break



        # plt.clf()
        # plt.xlabel('Generation')
        # plt.ylabel('Max Profit')
        # plt.title(f'Convergence Graph methods: {selection_methods}')
        # plt.plot(range(generation + 1), max_profits, label='Max Profit')
        # plt.pause(0.1)

    best_individual = max(population, key=calculate_fitness)
    best_fitness = calculate_fitness(best_individual)
    total_weight = sum(best_individual[i] * instruments[i][1] for i in range(len(best_individual)))
    print("\nGA found best solution after", generation, "iterations:")
    print("Expected profit =", best_fitness)
    print("Total weight =", total_weight)
    print("Included instruments are", [i for i in range(len(best_individual)) if best_individual[i] == 1])
    print('\n')
    plt.clf()
    plt.plot(range(generations), max_profits, label='Max Profit')
    window_size = 10
    smoothed_profits = moving_average(max_profits, window_size)

    plt.plot(range(window_size - 1, generations), smoothed_profits, label='Smoothed Max Profit', color='red')
    plt.xlabel('Generation')
    plt.ylabel('Max Profit')
    plt.title(f'Convergence Graph: {selection_methods}')
    plt.grid(True)
    plt.legend()
    plt.pause(3)
    return best_individual, best_fitness, total_weight

population = initialize_population(population_size, len(instruments))

# print(genetic_algorithm_maximize_the_profit('random',instruments,population, generations=1000, population_size=50, mutation_rate=0.75, crossover_rate=0.25))

# print(genetic_algorithm_maximize_the_profit('proportional',instruments,population, generations=1000, population_size=50))
# print(genetic_algorithm_maximize_the_profit('tournament',instruments,population, generations=1000, population_size=50))
# print(genetic_algorithm_maximize_the_profit('ranking',instruments,population, generations=1000, population_size=50))
# print(genetic_algorithm_maximize_the_profit('elite',instruments,population, generations=1000, population_size=50))
# print(genetic_algorithm_maximize_the_profit('progress_selection',instruments,population, generations=1000, population_size=50))
#





# Genetic Algorithm
def genetic_algorithm(selection_method):
    population = initialize_population(population_size, len(instruments))
    max_profits = []
    mutation_rate = np.random.random()
    crossover_rate = np.random.random()
    total_weight_list = []
    for generation in range(generations):





        mutation_rate = mutation_rate * 0.75
        crossover_rate = crossover_rate * 1.1
        fitnesses = calculate_fitnesses_fo_all_population(population)
        max_profits.append(max(fitnesses))
        if generation > 300 and max_profits[-1] == max_profits[-50] and  max_profits[-1] >= max(max_profits):
            for i in range(generation, generations-1):
                max_profits.append(max_profits[-1])
            break
        new_population = []
        total_weight_list.append(sum(max(population, key=calculate_fitness)[i] * instruments[i][1] for i in range(len(instruments))))
        if generation % 100 == 0 and generation > 0:

            print("Generation", generation, "profit:", max_profits[-1] if max_profits else 0,"total weight:", sum(max(population, key=calculate_fitness)[i] * instruments[i][1] for i in range(len(instruments))))
            print("Mutation rate:", mutation_rate, "Crossover rate:", crossover_rate)
            print("Best solution so far:", max(max_profits) if max_profits else 0, 'best_weight' , total_weight_list[max_profits.index(max(max_profits ))] if total_weight_list else 0)
            print('')

        for _ in range(population_size // 2):
            rand = np.random.random()
            if selection_method == 'roulette':
                parent1 = proportional_selection(population, fitnesses)
                parent2 = proportional_selection(population, fitnesses)

            elif selection_method == 'tournament':
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
            elif selection_method == 'ranking':
                parent1 = ranking_selection(population, fitnesses)
                parent2 = ranking_selection(population, fitnesses)
            elif selection_method == 'random':
                parent1 = random_selection(population)
                parent2 = random_selection(population)
            elif selection_method == 'mixed_random':


                if rand < 0.33:
                    parent1 = selection(population, fitnesses)
                    parent2 = selection(population, fitnesses)
                    print('selection')
                elif rand < 0.66 and rand >= 0.33:
                    parent1 = tournament_selection(population, fitnesses)
                    parent2 = tournament_selection(population, fitnesses)
                    print('tournament')
                else:
                    parent1 = ranking_selection(population, fitnesses)
                    parent2 = ranking_selection(population, fitnesses)
                    print('ranking')
            elif selection_method == 'progress_selection':
                if generations%3 == 0:
                    parent1 = selection(population, fitnesses)
                    parent2 = selection(population, fitnesses)
                elif generations%3 == 1:
                    parent1 = tournament_selection(population, fitnesses)
                    parent2 = tournament_selection(population, fitnesses)
                else:
                    parent1 = ranking_selection(population, fitnesses)
                    parent2 = ranking_selection(population, fitnesses)

            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])
        if mutation_rate < 0.01:
            mutation_rate = 0.9
        if crossover_rate > 0.9:
            crossover_rate = 0.1
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

def main():
    # print(genetic_algorithm_maximize_the_profit('random',instruments,population, generations=1000, population_size=50, mutation_rate=0.75, crossover_rate=0.25))
    methods = ['random', 'proportional', 'tournament', 'ranking', 'elite', 'progress_selection']
    results = []
    pop = initialize_population(population_size, len(instruments))
    for method in methods:
        print(f"Running GA with {method} selection\n")
        result = genetic_algorithm_maximize_the_profit(method, instruments, pop, generations=1000, population_size=50)
        print('\n')
        results.append(tuple([method, result[1], round(result[2], 2)]))
    print('the best solution is:', max(results, key=lambda x: x[1]))
    print('the best weight is:', min(results, key=lambda x: x[2]))
    print('\n')
    print('the worst solution is:', min(results, key=lambda x: x[1]))
    print('the worst weight is:', max(results, key=lambda x: x[2]))
    print('\n')




#Run the genetic algorithm with different selection methods
# print("Running GA with Roulette Wheel Selection\n")
# genetic_algorithm('roulette')
#
#
# print("\nRunning GA with Tournament Selection\n")
# genetic_algorithm('tournament')
#
# print("\nRunning GA with Ranking Selection\n")
# genetic_algorithm('ranking')
#
# print("\nRunning GA with Random Selection\n")
# genetic_algorithm('mixed_random')
#
# print("\nRunning GA with Progress Selection\n")
# genetic_algorithm('progress_selection')

# print("\nRunning GA with Random Selection\n")
# genetic_algorithm('random')


if __name__ == '__main__':
    main()
