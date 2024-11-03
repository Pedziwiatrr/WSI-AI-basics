import numpy as np
import random

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


def get_roulette_chances(cities_matrix, population):
    scores = []
    total_fitness_score = 0
    # calculate fitness score = 1/distance for every solution and add it to roulette wheel pool
    for solution in population:
        distance = evaluate_solution(cities_matrix, solution)
        assert distance > 0
        fitness_score = 1 / distance
        total_fitness_score += fitness_score
        scores.append(fitness_score)
    assert total_fitness_score > 0
    # return solutions probabilities based on their fitness score
    selection_chances = [score / total_fitness_score for score in scores]
    return selection_chances


def select_solution(cities_matrix, population, selection_chances):
    # select solution based on set probability
    return np.random.choice(population, p=selection_chances)


def crossover(cities_matrix, first_parent, second_parent):
    # select 2 indexes to set crossover points
    crossover_points = np.random.choice(range(1, len(first_parent)), 2, False)
    start_point, end_point = sorted(crossover_points)
    first_child = first_parent.copy()
    second_child = second_parent.copy()
    # swap the values between selected points in children
    first_middle_part = first_parent[start_point:end_point]
    second_middle_part = second_parent[start_point:end_point]
    first_child[start_point:end_point] = second_middle_part
    second_child[start_point:end_point] = first_middle_part
    # select the remaining parts with cities that would be missing after the swap
    first_remaining_part = [city for city in first_parent if city not in second_middle_part]
    second_remaining_part = [city for city in second_parent if city not in first_middle_part]
    # replace children's not-middle values with remaining cities to avoid duplicates and missing cities
    for i in range(0, len(first_parent)):
        if i not in range(start_point, end_point):
            first_child[i] = first_remaining_part.pop(0)
            second_child[i] = second_remaining_part.pop(0)
    return first_child, second_child


def generational_succession(cities_matrix, population):
    selection_chances = get_roulette_chances(cities_matrix, population)
    new_generation = []
    while len(new_generation) < len(population):
        # select parents using roulette selection algorithm
        first_parent = select_solution(cities_matrix, population, selection_chances)
        second_parent = select_solution(cities_matrix, population, selection_chances)
        # crossover or not based on set probability
        if random.uniform(0, 1) > CROSSOVER_PROBABILITY:
            first_child, second_child = first_parent, second_parent
        else:
            first_child, second_child = crossover(cities_matrix, first_parent, second_parent)
        first_child = mutate(first_child)
        second_child = mutate(second_child)
        # skip the children and notify about potential improper solutions
        try:
            validate_solution(first_child)
            validate_solution(second_child)
        except AssertionError:
            print("Incorrect solution detected!")
            continue
        new_generation.append(first_child)
        new_generation.append(second_child)
    return new_generation


def mutate(solution):
    swap_points = np.random.choice(range(1, len(solution)), 2, False)
    solution[swap_points[0]], solution[swap_points[1]] = solution[swap_points[1]], solution[swap_points[0]]
    return solution


def evolution_algorithm(cities_matrix, generation_count):
    population = generate_initial_population(cities_matrix, POPULATION_SIZE)
    best_solution = None
    shortest_length = float('inf')
    for generation_id in range(generation_count):
        new_population = generational_succession(cities_matrix, population)
        population = new_population
        for solution in population:
            solution_length = evaluate_solution(cities_matrix, solution)
            if solution_length < shortest_length:
                shortest_length = solution_length
                best_solution = solution
    return best_solution






