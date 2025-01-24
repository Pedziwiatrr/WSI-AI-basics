import argparse
from handle_data import get_numerical_data
from bayesian_network import create_network
from visualizer import visualize_network


def main():
    data = get_numerical_data()
    network = create_network(data)
    visualize_network(network)

if __name__ == '__main__':
    main()