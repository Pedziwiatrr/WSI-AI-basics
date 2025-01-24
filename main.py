import argparse
import matplotlib
import logging
from handle_data import get_prepared_data
from bayesian_network import create_network, get_feature_probability_distribution, get_feature_dependencies, get_marginal_probabilities
from visualizer import visualize_network, print_feature_dependencies, print_probabilities_distribution, plot_probability_distributions, print_marginal_probabilities
from generator import generate_murder


def main():
    matplotlib.use('Agg')
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--network_plot', action='store_true')
    parser.add_argument('--additional_plots', action='store_true')

    parser.add_argument('--print_distributions', action='store_true')
    parser.add_argument('--print_dependencies', action='store_true')
    parser.add_argument('--print_marginal_probabilities', action='store_true')

    parser.add_argument('--generate', action='store_true')

    parser.add_argument('--victim_sex', default='?', type=str)
    parser.add_argument('--victim_age', default='?', type=str)
    parser.add_argument('--victim_race', default='?', type=str)
    parser.add_argument('--perpetrator_sex', default='?', type=str)
    parser.add_argument('--perpetrator_age', default='?', type=str)
    parser.add_argument('--perpetrator_race', default='?', type=str)
    parser.add_argument('--relationship', default='?', type=str)
    parser.add_argument('--weapon', default='?', type=str)
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

    if args.generate:
        incomplete_murder_data = dict()
        incomplete_murder_data['Victim Sex'] = args.victim_sex
        incomplete_murder_data['Victim Age'] = args.victim_age
        incomplete_murder_data['Victim Race'] = args.victim_race
        incomplete_murder_data['Perpetrator Sex'] = args.perpetrator_sex
        incomplete_murder_data['Perpetrator Age'] = args.perpetrator_age
        incomplete_murder_data['Perpetrator Race'] = args.perpetrator_race
        incomplete_murder_data['Relationship'] = args.relationship
        incomplete_murder_data['Weapon'] = args.weapon
        print(f"\nINPUT MURDER DATA:\n {incomplete_murder_data}\n")
        generated_murder = generate_murder(distributions_dict, marginal_probabilities_dict, incomplete_murder_data)
        print(f"OUTPUT MURDER DATA:\n {generated_murder}\n")


if __name__ == '__main__':
    main()