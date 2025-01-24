import argparse
import matplotlib
from handle_data import get_numerical_data, get_prepared_data
from bayesian_network import create_network, get_feature_probability_distribution, get_feature_dependencies, get_marginal_probabilities
from visualizer import visualize_network, print_feature_dependencies, print_probabilities_distribution, plot_probability_distributions, print_marginal_probabilities
from generator import generate_murder


INCOMPLETE_MURDER_DATA = {
    'Victim Sex': '?',
    'Victim Age': '20',
    'Victim Race': '?',
    'Perpetrator Sex': 'male',
    'Perpetrator Age': '?',
    'Perpetrator Race': 'asian',
    'Relationship': 'friend',
    'Weapon': 'strangulation'
}


def main():
    matplotlib.use('Agg')

    parser = argparse.ArgumentParser()
    parser.add_argument('--network_plot', action='store_true')
    parser.add_argument('--additional_plots', action='store_true')
    parser.add_argument('--print_distributions', action='store_true')
    parser.add_argument('--print_dependencies', action='store_true')
    parser.add_argument('--print_marginal_probabilities', action='store_true')
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    
    data = get_prepared_data()
    network = create_network(data)
    if args.network_plot:
        visualize_network(network)

    dependencies_dict = get_feature_dependencies(network)
    distributions_dict = get_feature_probability_distribution(network)
    marginal_probabilities_dict = get_marginal_probabilities(data)

    if args.print_distributions:
        print_probabilities_distribution(distributions_dict)
    if args.print_dependencies:
        print_feature_dependencies(dependencies_dict)
    if args.print_marginal_probabilities:
        print_marginal_probabilities(marginal_probabilities_dict)
    if args.additional_plots:
        plot_probability_distributions(marginal_probabilities_dict)
    if args.complete:
        generated_murder = generate_murder(dependencies_dict, marginal_probabilities_dict, INCOMPLETE_MURDER_DATA)
        print(generated_murder)


if __name__ == '__main__':
    main()