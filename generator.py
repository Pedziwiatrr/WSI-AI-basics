import random
import numpy as np


def generate_murder(probability_distribution, marginal_probabilities, murder):

    missing_features = [feature for feature in probability_distribution.keys()
                        if murder[feature] == '?']

    for feature in missing_features:
        cpd = probability_distribution[feature]


        if len(cpd.variables) == 1:
        # len == 1 -> no dependencies, it is a marginal probability
            probabilities = marginal_probabilities[feature]
            # choose the value basing on stated probabilities
            value = random.choices(list(probabilities.keys()), weights=list(probabilities.values()))[0]
        else:
        # len > 1 -> this feature depends on others
            pass

        murder[feature] = value

    return murder