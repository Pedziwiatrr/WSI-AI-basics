from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator
from collections import defaultdict


def create_network(data):
    # choose the best network's structure using BicScore as a criterion
    quality_rating = BicScore(data)
    hill_climb_search = HillClimbSearch(data)
    best_network = hill_climb_search.estimate(quality_rating)

    # use the best structure to create our network
    network = BayesianNetwork(best_network.edges())
    # estimate conditional probabilities for different data occurrences
    network.fit(data, estimator=MaximumLikelihoodEstimator)

    return network


def get_feature_probability_distribution(network):
    probability_distribution = {}
    for feature in network.nodes():
        distribution = network.get_cpds(feature)
        probability_distribution[feature] = distribution

    return probability_distribution


def get_feature_dependencies(network):
    dependencies = defaultdict(list)
    for edge in network.edges():
        dependencies[edge[0]].append((edge[1]))
        #print(f"edge: {edge[0]} -> {edge[1]}")
    #print(f"\ndependencies: {dependencies}\n")
    return dependencies
