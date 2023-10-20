import math
import numpy as np
from scipy.special import gamma                         # scipy gamma function is better than in math


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

    alpha = 1       # shape minimum 2 so that it starts by 0
    beta = 1      # rate

    increment = np.random.gamma(alpha*t, beta)

    return increment


def pavement_deterioration_variance_gamma_process(t):

    alpha = 2       # shape minimum 2 so that it starts by 0
    beta = 1      # rate

    deterioration_delta = np.random.gamma(alpha*t, beta) - np.random.gamma(alpha*t, beta)     # difference bewtween two independent gamma processes

    return deterioration_delta


def pavement_deterioration_random_process(age):

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
