import random
import numpy as np

def generate_murder(distributions_dict, marginal_probabilities_dict, murder_data):
    for feature, value in murder_data.items():
        if value == '?':
            if feature in distributions_dict:
                print(f"\nGENERATING FEATURE: {feature}")
                # Use the conditional probability distribution
                distribution = distributions_dict[feature]

                # Find the probabilities for the feature based on known parents (feature's influencers)
                feature_index = distribution.variables.index(feature)
                parent_values = {parent: murder_data[parent] for parent in distribution.variables if parent != feature}
                print(f"parent_values: {parent_values}")

                # Get all values probabilities
                possible_values = distribution.state_names[feature]
                probabilities = distribution.values
                print(f"Type of probabilities: {type(probabilities)}")
                print(f"Shape of probabilities: {probabilities.shape}")

                if probabilities.ndim > 1:
                    # Iterating in reverse so the previous indexes won't change
                    for parent, parent_value in reversed(list(parent_values.items())):
                        if parent_value != '?':
                            parent_index = distribution.variables.index(parent)
                            state_index = distribution.state_names[parent].index(parent_value)
                            probabilities = np.take(probabilities, state_index, axis=parent_index)
                            print(f"parent_index: {parent_index}, parent_value: {parent_value}, state_index: {state_index}, probabilities dim: {probabilities.ndim}")

                combined_probabilities = probabilities / np.sum(probabilities)
                print(f"Normalized probabilities: {combined_probabilities}")

                # Roulette selection like in evolution algorithm
                murder_data[feature] = random.choices(possible_values, weights=combined_probabilities, k=1)[0]
            else:
                probabilities = marginal_probabilities_dict[feature]
                possible_values = list(probabilities.keys())
                weights = list(probabilities.values())

                murder_data[feature] = random.choices(possible_values, weights=weights, k=1)[0]

    return murder_data
