import matplotlib.pyplot as plt
import networkx as nx


def visualize_network(network):
    graph = nx.DiGraph()
    for edge in network.edges():
        graph.add_edge(edge[0], edge[1])

    pos = nx.spring_layout(graph, k=5, iterations=50)
    nx.draw(graph, pos, with_labels=True, node_size=2500, node_color="skyblue", font_size=10, font_weight="bold", arrows=True)

    plt.title("Bayesian Network Visualization")
    plt.savefig("bayesian_network.png")
    plt.close()


def print_feature_dependencies(dependencies_dict):
    print("-"*100)
    for feature, dependencies in dependencies_dict.items():
        if dependencies:
            print(f"> {feature}: is influenced by:")
            for dependency in dependencies:
                print(f"    <- {dependency}")


def print_probabilities_distribution(distributions_dict):
    print("-"*100)
    for feature, distribution in distributions_dict.items():
        print(f"> {feature}'s distribution:")
        print(f"{distribution}")


def plot_probability_distributions(marginal_probabilities):
    for feature, probabilities in marginal_probabilities.items():
        if feature in ['Victim Age', 'Perpetrator Age']:
            probabilities = {
                value: prob for value, prob in probabilities.items()
                if 1 <= int(value) < 99
            }
            probabilities = dict(sorted(probabilities.items(), key=lambda x: int(x[0])))

        values = list(probabilities.keys())
        probs = list(probabilities.values())

        plt.figure(figsize=(20, 10))
        plt.bar(values, probs, alpha=0.7)
        plt.title(f'Probability Distribution of {feature}')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        if feature == 'Victim Age' or feature == 'Perpetrator Age':
            plt.xticks(values, fontsize=8)
        else:
            plt.xticks(values, rotation=45, fontsize=20)
        plt.tight_layout()
        plt.savefig(f'plots/probability_distribution_plot_of_{feature.lower()}.png')
        plt.close()


def print_marginal_probabilities(marginal_probabilities):
    for feature, probabilities in marginal_probabilities.items():
        print("-" * 100)
        print(f"\n> Feature: {feature}")
        for value, probability in probabilities.items():
            print(f"    {value}: {probability:.4f}")
