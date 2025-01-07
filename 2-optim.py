from funcs import *
from consts import *
import numpy as np
import logging

random.seed(0)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.disable(logging.CRITICAL)

X2_GRADATION = 4

def decimal_to_two_values(decimal):
    x1 = decimal // X2_GRADATION
    x2 = decimal % X2_GRADATION
    return (x1, x2)


def f2(x1, x2):
    return np.cos(x1 * x2) * np.sin(x1) * (x1 + x2)


def main(gray=False):
    if gray:
        decimal_to_code = lambda x, length: binary_to_gray(decimal_to_binary(x, length))
        code_to_decimal = lambda x: decimal_to_two_values(binary_to_decimal(gray_to_binary(x)))
    else:
        decimal_to_code = lambda x, length: decimal_to_binary(x, length)
        code_to_decimal = lambda x: decimal_to_two_values(binary_to_decimal(x))

    # Initial population
    population = [decimal_to_code(random.randint(0, 2 ** GENE_LENGTH - 1), GENE_LENGTH) for _ in range(POPULATION_SIZE)]
    # Evaluate initial fitness
    fitness_values = [f2(*code_to_decimal(individual)) for individual in population]
    logging.debug(f"Initial population: {population}")
    logging.debug(f"Initial population: {[code_to_decimal(individual) for individual in population]}")
    logging.debug(f"Initial fitness values: {fitness_values}")
    logging.debug("")

    for _ in range(GENERATIONS):
        # Pick elite individuals
        elite_indices = np.argsort(fitness_values)[-ELITE_COUNT:]
        # pop elite individuals from population to new_population
        new_population = [population[i] for i in elite_indices]
        population = [population[i] for i in range(len(population)) if i not in elite_indices]
        fitness_values = [fitness_values[i] for i in range(len(fitness_values)) if i not in elite_indices]

        # Selection
        population = roulette_selection(population, fitness_values)
        logging.debug(f"Selected population: {population}")
        logging.debug(f"Selected population: {[code_to_decimal(individual) for individual in population]}")
        logging.debug("")
        # Crossover
        logging.debug("Crossover")
        population = crossover(population, CROSSOVER_RATE)
        logging.debug(f"Crossover population: {population}")
        logging.debug(f"Crossover population: {[code_to_decimal(individual) for individual in population]}")
        logging.debug("")
        # Mutation
        population = [mutate(individual, MUTATION_RATE) for individual in population]
        logging.debug(f"Mutated population: {population}")
        logging.debug(f"Mutated population: {[code_to_decimal(individual) for individual in population]}")
        logging.debug("")

        new_population.extend(population)
        population = new_population
        # Evaluate fitness
        fitness_values = [f2(*code_to_decimal(individual)) for individual in population]
        logging.debug(f"new population: {population}")
        logging.debug(f"new population: {[code_to_decimal(individual) for individual in population]}")
        logging.debug(f"new fitness values: {fitness_values}")
        logging.debug("")

    # Final result
    best_individual = population[np.argmax(fitness_values)]
    best_decimal = code_to_decimal(best_individual)
    best_fitness = f2(*best_decimal)

    print(f"Best individual: {best_individual}")
    print(f"Best decimal value: {best_decimal}")
    print(f"Best fitness value: {best_fitness}")


if __name__ == "__main__":
    print("Binary encoding")
    main()
    print()
    print("Gray encoding")
    main(gray=True)
