from pgmpy.models import BayesianNetwork
from pgmpy.estimators import HillClimbSearch, BicScore, MaximumLikelihoodEstimator


def create_network(data):
    # choose the best network's structure using BicScore as a criterion
    quality_rating = BicScore(data)
    hill_climb_search = HillClimbSearch(data)
    best_network = hill_climb_search.estimate(quality_rating)

    # use the best structure to create our network
    network = BayesianNetwork(best_network.edges())
    network.fit(data, estimator=MaximumLikelihoodEstimator)

    return network