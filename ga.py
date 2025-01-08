import numpy as np
import logging
import random

POPULATION_SIZE = 5
GENERATIONS = 3
ELITE_COUNT = 1
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.05
NORMALIZE_MIN_RATIO = 0.1

def normalize_fitness(fitness_values, min_ratio=NORMALIZE_MIN_RATIO):
    min_fitness = min(fitness_values)
    max_fitness = max(fitness_values)
    epsilon = min_ratio * (max_fitness - min_fitness)
    if epsilon == 0:
        return list(np.abs(fitness_values) + 1)
    return [a + (epsilon - min_fitness) for a in fitness_values]


def roulette_selection(population, fitness_values, minimize=False):
    if minimize:
        fitness_values = [-a for a in fitness_values]
    fitness_values = normalize_fitness(fitness_values)
    return random.choices(
        population,
        weights=[a / sum(fitness_values) for a in fitness_values],
        k=len(population)
    )


def crossover(population, crossover_rate):
    for i in range(len(population) - 1):
        for j in range(i + 1, len(population)):
            if random.random() < crossover_rate:
                point = random.randint(1, len(population[i]) - 1)
                logging.debug(f"{population[i]} and {population[j]} (point:{point})")
                child1 = population[i][:point] + population[j][point:]
                child2 = population[j][:point] + population[i][point:]
                population[i] = child1
                population[j] = child2
                logging.debug(f"  result: {population[i]}, {population[j]}")
    return population


def GA(fitness_func, gtype_to_ptype, initial_population, mutate, minimize=False):
    # Initial population
    population = [initial_population() for _ in range(POPULATION_SIZE)]
    # Evaluate initial fitness
    fitness_values = [fitness_func(gtype_to_ptype(individual)) for individual in population]
    logging.debug(f"Initial population(gtype): {population}")
    logging.debug(f"Initial population(ptype): {[gtype_to_ptype(individual) for individual in population]}")
    logging.debug(f"Initial fitness values: {fitness_values}")
    logging.debug("")

    for _ in range(GENERATIONS):
        # Pick elite individuals
        elite_indices = np.argsort(fitness_values)[-ELITE_COUNT:] if not minimize else np.argsort(fitness_values)[:ELITE_COUNT]
        # pop elite individuals from population to new_population
        new_population = [population[i] for i in elite_indices]
        population = [population[i] for i in range(len(population)) if i not in elite_indices]
        fitness_values = [fitness_values[i] for i in range(len(fitness_values)) if i not in elite_indices]

        # Selection
        population = roulette_selection(population, fitness_values, minimize)
        logging.debug(f"Selected population(gtype): {population}")
        logging.debug(f"Selected population(ptype): {[gtype_to_ptype(individual) for individual in population]}")
        logging.debug("")
        # Crossover
        logging.debug("Crossover")
        population = crossover(population, CROSSOVER_RATE)
        logging.debug(f"Crossover population(gtype): {population}")
        logging.debug(f"Crossover population(ptype): {[gtype_to_ptype(individual) for individual in population]}")
        logging.debug("")
        # Mutation
        population = [mutate(individual, MUTATION_RATE) for individual in population]
        logging.debug(f"Mutated population(gtype): {population}")
        logging.debug(f"Mutated population(ptype): {[gtype_to_ptype(individual) for individual in population]}")
        logging.debug("")

        new_population.extend(population)
        population = new_population
        # Evaluate fitness
        fitness_values = [fitness_func(gtype_to_ptype(individual)) for individual in population]
        logging.debug(f"new population(gtype): {population}")
        logging.debug(f"new population(ptype): {[gtype_to_ptype(individual) for individual in population]}")
        logging.debug(f"new fitness values: {fitness_values}")
        logging.debug("")

    # Final result
    best_individual = population[np.argmax(fitness_values)] if not minimize else population[np.argmin(fitness_values)]
    best_ptype = gtype_to_ptype(best_individual)
    best_fitness = fitness_func(best_ptype)

    print(f"Best individual: {best_individual}")
    print(f"Best ptype value: {best_ptype}")
    print(f"Best fitness value: {best_fitness}")
