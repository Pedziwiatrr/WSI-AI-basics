import argparse
import pathlib
import time

import numpy as np
import pandas as pd
from evolution import decode_solution, evolution_algorithm
from mapper import create_map

MINI_CITIES_NUM = 5
CROSSOVER_PROBABILITY = 0.75
MUTATION_PROBABILITY = 0.5


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities-path", type=pathlib.Path, required=True, help="Path to cities csv file")
    parser.add_argument(
        "--problem-size",
        choices=["mini", "full"],
        default="mini",
        help="Run algorithm on full or simplified problem setup",
    )
    parser.add_argument("--start", type=str, default="Skierniewice")
    parser.add_argument("--finish", type=str, default="Warszawa")
    parser.add_argument("--pop_size", type=int, default=250)
    parser.add_argument("--gen_count", type=int, default=1000)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--map", action="store_true", help="Flag to generate map visualization")
    return parser.parse_args()


def load_data(args):
    data = pd.read_csv(args.cities_path)
    data_without_start_and_finish = data[~((data.index == args.finish) | (data.index == args.start))]
    if args.problem_size == "mini":
        city_names = (
            [args.start] + data_without_start_and_finish.sample(n=MINI_CITIES_NUM - 2).index.tolist() + [args.finish]
        )
    else:
        city_names = [args.start] + data_without_start_and_finish.index.tolist() + [args.finish]

    return data[city_names].loc[city_names]


def main():
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    data = load_data(args)

    start_time = time.time()
    solution, length = evolution_algorithm(data, args.gen_count, args.pop_size, CROSSOVER_PROBABILITY, MUTATION_PROBABILITY)
    end_time = time.time()
    total_time = end_time - start_time

    print("\n" + "="*100)
    print("Best found solution: " + str(decode_solution(data, solution)))
    print("\nDistance: " + str(round(length, 3)) + "km")
    print("Execution time: " + str(total_time) + "s")
    print("Generation count: " + str(args.gen_count) + "\nPopulation size: " + str(args.pop_size))
    print( "\nCrossover probability: " + str(CROSSOVER_PROBABILITY*100) + "%\nMutation probability: " + str(MUTATION_PROBABILITY*100) + "%")
    print("")
    print("="*100 + "\n")
    if args.map:
        create_map(decode_solution(data, solution))


if __name__ == "__main__":
    main()
