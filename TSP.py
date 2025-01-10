from ga import GA
import numpy as np
import logging
import random

random.seed(0)
np.random.seed(0)
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logging.disable(logging.CRITICAL)

CITIES = np.array([
    [0, 0],  # 都市1
    [6, 8],  # 都市2
    [3, 10],  # 都市3
    [2, 5],  # 都市4
    [3, 3]   # 都市5
])


def eval_cost(route):
    distance = 0
    for i in range(len(route)):  # 出発点へ帰る
        dest = CITIES[route[i]]
        source = CITIES[route[i - 1]]
        distance += np.linalg.norm(dest - source)
    return distance


def mutate_TSP(route, mutation_rate):
    '''
    codeがint型の場合のmutation
    '''
    for i in range(len(route)):
        if random.random() < mutation_rate:
            possible_value = list(range(len(route) - i))
            possible_value.remove(route[i])
            if len(possible_value) == 0:
                continue
            route[i] = random.choice(possible_value)
    return route


def code_to_route(code):
    route = []
    order = list(range(len(CITIES)))
    for c in code:
        route.append(order[c])
        order.remove(order[c])
    return route


def route_to_code(route):
    code = []
    order = list(range(len(CITIES)))
    for r in route:
        code.append(order.index(r))
        order.remove(r)
    return code


if __name__ == "__main__":
    print("TSP")
    print()
    ptype_to_gtype = lambda x: route_to_code(x)
    gtype_to_ptype = lambda x: code_to_route(x)
    initial_population = lambda: ptype_to_gtype(np.random.permutation(len(CITIES)))
    GA(eval_cost, gtype_to_ptype, initial_population, mutate_TSP, minimize=True)
