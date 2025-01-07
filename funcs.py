import random
from consts import *
import logging
import numpy as np

def binary_to_gray(binary):
    gray = binary[0]
    for i in range(1, len(binary)):
        gray += str(int(binary[i - 1]) ^ int(binary[i]))
    return gray


def gray_to_binary(gray):
    binary = gray[0]
    for i in range(1, len(gray)):
        binary += str(int(binary[-1]) ^ int(gray[i]))
    return binary


def binary_to_decimal(binary):
    return int(binary, 2)


def decimal_to_binary(decimal, length):
    return bin(decimal)[2:].zfill(length)


def normalize_fitness(fitness_values, min_ratio=NORMALIZE_MIN_RATIO):
    min_fitness = min(fitness_values)
    max_fitness = max(fitness_values)
    epsilon = min_ratio * (max_fitness - min_fitness)
    if epsilon == 0:
        return list(np.abs(fitness_values) + 1)
    return [a + (epsilon - min_fitness) for a in fitness_values]


def roulette_selection(population, fitness_values):
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

def mutate(individual, mutation_rate):
    return ''.join(
        str(1 - int(bit)) if random.random() < mutation_rate else bit
        for bit in individual
    )
