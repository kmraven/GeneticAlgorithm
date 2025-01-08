from ga import GA
import numpy as np
import logging
import random

random.seed(0)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.disable(logging.CRITICAL)

GENE_LENGTH = 5
X2_GRADATION = 4

def f1(x):
    return 1000 * np.sin(x/2) - x**2


def f2(x):
    x1, x2 = x
    return np.cos(x1 * x2) * np.sin(x1) * (x1 + x2)


def mutate_optim(individual, mutation_rate):
    '''
    codeがbit(str)型のmutation
    '''
    return ''.join(
        str(1 - int(bit)) if random.random() < mutation_rate else bit
        for bit in individual
    )


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


def decimal_to_binary(decimal, length=GENE_LENGTH):
    return bin(decimal)[2:].zfill(length)


def decimal_to_two_values(decimal):
    x1 = decimal // X2_GRADATION
    x2 = decimal % X2_GRADATION
    return (x1, x2)


if __name__ == "__main__":
    print("1 Variable Optimization")
    print()
    print("Binary encoding")
    ptype_to_gtype = lambda x: decimal_to_binary(x)
    gtype_to_ptype = lambda x: binary_to_decimal(x)
    initial_population = lambda: ptype_to_gtype(random.randint(0, 2 ** GENE_LENGTH - 1))
    GA(f1, gtype_to_ptype, initial_population, mutate_optim)
    print()
    print("Gray encoding")
    ptype_to_gtype = lambda x: binary_to_gray(decimal_to_binary(x))
    gtype_to_ptype = lambda x: binary_to_decimal(gray_to_binary(x))
    GA(f1, gtype_to_ptype, initial_population, mutate_optim)

    print()

    print("2 Variable Optimization")
    print()
    print("Binary encoding")
    ptype_to_gtype = lambda x: decimal_to_binary(x)
    gtype_to_ptype = lambda x: decimal_to_two_values(binary_to_decimal(x))
    GA(f2, gtype_to_ptype, initial_population, mutate_optim)
    print()
    print("Gray encoding")
    ptype_to_gtype = lambda x: binary_to_gray(decimal_to_binary(x))
    gtype_to_ptype = lambda x: decimal_to_two_values(binary_to_decimal(gray_to_binary(x)))
    GA(f2, gtype_to_ptype, initial_population, mutate_optim)
