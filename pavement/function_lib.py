import numpy as np


def pavement_deterioration(pci, pci_groups, transition_matrix, status, t):

    # Markov Chain after t time steps
    p_distribution = status.dot(np.linalg.matrix_power(transition_matrix, t))                # probability distribution at t

    # MC samples
    sample = np.random.choice(pci_groups, 10000, p=p_distribution)

    # Expected value (arithmetic mean)
    mean = np.mean(sample)

    return mean
