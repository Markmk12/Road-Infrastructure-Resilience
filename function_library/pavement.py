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


def pavement_deterioration_gamma_process(pci, t):

    # Gamma distribution
    shape_k = 5
    scale_theta = 1
    mean = shape_k * scale_theta
    variance = shape_k * pow(scale_theta, 2)

    # Gamma process parameterised in terms of the mean and variance
    rate_gamma = pow(mean, 2) / variance
    rate_lambda = mean / variance

    # Gamma process
    pci_degradation = (pow(rate_lambda, rate_gamma * t) / gamma(rate_gamma * t)) * pow(pci, rate_gamma * t - 1) * np.exp(-rate_lambda * pci)

    return pci_degradation


def pavement_deterioration_gamma_process_alternative(pci, t):

    alpha = 2       # shape minimum 2 so that it starts by 0
    beta = 1      # rate

    # Zeitintervall für jeden Schritt
    #dt = t / 100

    # Generiere unabhängige Gamma-verteilte Zuwächse
    #increments = np.random.gamma(alpha * dt, 1 / beta, 100)
    increment = np.random.gamma(alpha*t, beta)

    # Akkumuliere die Zuwächse, um den Prozess zu konstruieren
    #pci_degradation = np.sum(increments)

    return increment


def pavement_deterioration_variance_gamma_process(pci, t):

    alpha = 2       # shape minimum 2 so that it starts by 0
    beta = 1      # rate

    increment = np.random.gamma(alpha*t, beta) - np.random.gamma(alpha*t, beta)     # difference bewtween two independent gamma processes

    return increment