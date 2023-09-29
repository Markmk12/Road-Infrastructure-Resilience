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

    # Gamma distribution
    shape_k = 2
    scale_theta = 2
    mean = shape_k * scale_theta
    variance = shape_k * pow(scale_theta, 2)

    # Gamma process parameterised in terms of the mean and variance
    rate_gamma = pow(mean, 2) / variance
    rate_lambda = mean / variance

    # Gamma process
    pci_sample = (pow(rate_lambda, rate_gamma*t)/math.gamma(rate_gamma*t)) * pow(pci, rate_gamma*t-1) * np.exp(-rate_lambda*pci)

    return pci_sample
