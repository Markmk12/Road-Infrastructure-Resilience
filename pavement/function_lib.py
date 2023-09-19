import numpy as np


def pavement_deterioration(PCI, transition_matrix, status, t):

    # probability distribution at t
    v = status.dot(np.linalg.matrix_power(transition_matrix, t))

    # MC-Simulation


    degraded_PCI = PCI - 0.99
    return degraded_PCI

def mcmc_pavement(pci):
    a = pci+1
    return a
