import argparse
from handle_data import get_numerical_data
from bayesian_network import create_network, get_feature_probability_distribution, get_feature_dependencies
from visualizer import visualize_network, visualize_feature_dependencies, visualize_probabilities_distribution


def main():
    data = get_numerical_data()
    network = create_network(data)
    visualize_network(network)

    dependencies_dict = get_feature_dependencies(network)
    visualize_feature_dependencies(dependencies_dict)

    distributions_dict = get_feature_probability_distribution(network)
    visualize_probabilities_distribution(distributions_dict)

if __name__ == '__main__':
    main()