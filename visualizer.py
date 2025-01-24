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
        print(f"> {feature}'s distribution: {distribution}")

