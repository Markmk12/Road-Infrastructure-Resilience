import numpy as np


def pavement_deterioration_random_process(age):
    """
    Calculate the deterioration delta of pavement based on age using a random process.

    The deterioration is modeled as a combination of environmental and traffic effects,
    each represented by a gamma distribution. The total deterioration is a weighted
    sum of the environmental and traffic-induced deterioration. The random process is
    a linear combination of two independent gamma processes.

    Parameters:
    -----------
    age : float
        Age of the pavement. Negative values are corrected to zero.

    Returns:
    --------
    float
        Deterioration delta value representing the combined environmental and traffic-induced effects.

    Notes:
    ------
    1. The environmental deterioration parameters `alpha_environment` and `beta_environment`
       are currently hardcoded as 1.
    2. The traffic deterioration parameters `alpha_traffic` and `beta_traffic` are
       currently hardcoded as 0.5 and 0.3, respectively.
    3. Weights for environmental and traffic effects (`weight_environment` and `weight_traffic`)
       are hardcoded as 0.35 and 0.65, respectively.

    """

    # Logical correction
    if age < 0:
        age = 0

    # Environmental deterioration
    alpha_environment = 1
    beta_environment = 1

    # Deterioration through traffic
    alpha_traffic = 0.5
    beta_traffic = 0.3

    # Weight of the degradation
    weight_environment = 0.35
    weight_traffic = 0.65

    deterioration_delta = weight_environment * np.random.gamma(alpha_environment * age, beta_environment) + weight_traffic * np.random.gamma(alpha_traffic * age, beta_traffic)

    return deterioration_delta


# The following functions are alternative random processes for degradation modeling.
# They are not used in the code and may not work.

def pavement_deterioration_markov_chain(pci, pci_groups, transition_matrix, status, t):

    # Markov Chain after t time steps
    p_distribution = status.dot(np.linalg.matrix_power(transition_matrix, t))           # probability distribution at t

    # MC samples
    sample = np.random.choice(pci_groups, 100000, p=p_distribution)            # best results when using >10,000 samples

    # Expected value (arithmetic mean)
    mean = np.mean(sample)

    return mean


def pavement_deterioration_markov_chain_alternative(pci, pci_groups, transition_matrix, status, t):

    # Markov Chain after t time steps
    p_distribution = status.dot(np.linalg.matrix_power(transition_matrix, t))           # probability distribution at t

    # MC samples
    sample = np.random.choice(pci_groups, 1, p=p_distribution)            # best results when using >10,000 samples

    return sample


def pavement_deterioration_gamma_process(t):

    alpha = 1
    beta = 1

    increment = np.random.gamma(alpha*t, beta)

    return increment


def pavement_deterioration_variance_gamma_process(t):

    alpha = 2
    beta = 1

    # Difference between two independent gamma processes
    deterioration_delta = np.random.gamma(alpha*t, beta) - np.random.gamma(alpha*t, beta)

    return deterioration_delta
