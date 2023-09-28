import math
import numpy as np


def pavement_deterioration_markov_chain(pci, pci_groups, transition_matrix, status, t):

    # Markov Chain after t time steps
    p_distribution = status.dot(np.linalg.matrix_power(transition_matrix, t))           # probability distribution at t

    # MC samples
    sample = np.random.choice(pci_groups, 100000, p=p_distribution)            # best results when using >10,000 samples

    # Expected value (arithmetic mean)
    mean = np.mean(sample)

    return mean


def pavement_deterioration_gamma_process(pci, t):

    pci = math.gamma(t)

    return pci
