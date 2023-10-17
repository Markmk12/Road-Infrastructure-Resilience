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


def pavement_deterioration_random_process(t):

    # Logical correction
    if t < 0:
        t = 0

    # Parameter environment (natural deterioration???)
    alpha_environment = 1       # shape minimum 2 so that it starts by 0
    beta_environment = 1      # rate

    # Parameter traffic load
    alpha_traffic = 0.5
    beta_traffic = 0.5

    deterioration_delta = np.random.gamma(alpha_environment*t, beta_environment) + np.random.gamma(alpha_traffic*t, beta_traffic)

    return deterioration_delta
