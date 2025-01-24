import argparse
from handle_data import get_numerical_data
from bayesian_network import create_network, get_feature_probability_distribution, get_feature_dependencies
from visualizer import visualize_network, print_feature_dependencies, print_probabilities_distribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_plot', action='store_true')
    parser.add_argument('--print', action='store_true')
    args = parser.parse_args()

    data = get_numerical_data()
    network = create_network(data)
    if args.network_plot:
        visualize_network(network)

    dependencies_dict = get_feature_dependencies(network)
    distributions_dict = get_feature_probability_distribution(network)
    if args.additional_prints:
        print_probabilities_distribution(distributions_dict)
        print_feature_dependencies(dependencies_dict)


if __name__ == '__main__':
    main()