import math
import numpy as np
from scipy.special import gamma                         # scipy gamma function is better than in math
import matplotlib.pyplot as plt


def pavement_deterioration_gamma_process(pci, t):

    # Gamma distribution
    shape_k = 1
    scale_theta = 2
    mean = shape_k * scale_theta
    variance = shape_k * pow(scale_theta, 2)

    # Gamma process parameterised in terms of the mean and variance
    rate_gamma = pow(mean, 2) / variance
    rate_lambda = mean / variance

    # Gamma process
    pci_degradation = (pow(rate_lambda, rate_gamma * t) / gamma(rate_gamma * t)) * pow(pci, rate_gamma * t - 1) * np.exp(-rate_lambda * pci)

    return pci_degradation


time = range(10)

gamma_values = []

for t in time:
    gamma_value = pavement_deterioration_gamma_process(1, t)
    gamma_values.append(gamma_value)

print(gamma_values)

# Zeitwerte
t_values = list(time)

# Plot
plt.plot(t_values, gamma_values, marker='o', linestyle='-', color='b')
plt.xlabel('Time (t)')
plt.ylabel('Gamma Value')
plt.title('Gamma Values over Time')
plt.grid(True)
plt.show()
