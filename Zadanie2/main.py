import argparse
import pathlib

import numpy as np
import pandas as pd
from evolution import decode_solution, evolution_algorithm

MINI_CITIES_NUM = 5
ITERATIONS = 50


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
    parser.add_argument("--seed", type=int)
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

    solution, length = evolution_algorithm(data, ITERATIONS)
    print("\n" + "="*100)
    print("Best found solution: " + str(decode_solution(data, solution)))
    print("\nDistance: " + str(round(length, 3)) + "km")
    print("="*100 + "\n")


if __name__ == "__main__":
    main()
