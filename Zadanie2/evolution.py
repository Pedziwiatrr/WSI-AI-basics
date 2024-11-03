import numpy as np


POPULATION_SIZE = 1000
CROSSOVER_PROBABILITY = 0.5


def decode_solution(cities_matrix, solution):
    # return cities names using their indexes in solution
    return list(map(lambda city_id: cities_matrix.index[city_id], solution))


def validate_solution(cities_matrix, solution):
    # check if each city is visited exactly one time
    assert len(list(solution)) == len(set(solution))
    assert sorted(solution) == list(range(len(cities_matrix)))
    # check if start and finish cities are in the correct place
    assert solution[0] == 0 and solution[-1] == len(cities_matrix) - 1


def evaluate_solution(cities_matrix, solution):
    total_distance = 0
    for city_id in range(len(solution) - 1):
        total_distance += cities_matrix.iloc[solution[city_id], solution[city_id + 1]]
    return total_distance


def generate_solution(cities_matrix):
    return [0] + np.random.permutation(np.arange(1, len(cities_matrix) - 1)).tolist() + [len(cities_matrix) - 1]


def generate_initial_population(cities_matrix, size):
    population = []
    for i in range(size):
        population.append(generate_solution(cities_matrix))
    return population


def select_solutions(cities_matrix, population):
    scores = []
    total_fitness_score = 0
    # calculate fitness score = 1/distance for every solution and add it to roulette wheel pool
    for solution in population:
        distance = evaluate_solution(cities_matrix, solution)
        assert distance > 0
        fitness_score = 1 / distance
        total_fitness_score += fitness_score
        scores.append(fitness_score)
    # select part of solutions with probability depending on their fitness scores
    assert total_fitness_score > 0
    selection_chance = [score / total_fitness_score for score in scores]
    selected = np.random.choice(population, POPULATION_SIZE * CROSSOVER_PROBABILITY, p=selection_chance)
    return selected


def two_point_crossover(cities_matrix, first_parent, second_parent):
    # select 2 indexes to set crossover points
    crossover_points = np.random.choice(range(1, len(first_parent)), 2, False)
    start_point = crossover_points[0]
    end_point = crossover_points[1]
    # place points in order
    if start_point > end_point:
        temp = end_point
        end_point = start_point
        start_point = temp
    # copy parent solutions
    first_child = first_parent.copy()
    second_child = second_parent.copy()
    # swap parts between selected points
    first_child[start_point:end_point] = second_parent[start_point:end_point]
    second_child[start_point:end_point] = first_parent[start_point:end_point]

    try:
        validate_solution(cities_matrix, first_child)
        validate_solution(cities_matrix, second_child)
    except AssertionError:
        pass
        # here will be the part where solutions are being "fixed"


def mutate(solution):
    pass







