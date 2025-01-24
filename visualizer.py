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