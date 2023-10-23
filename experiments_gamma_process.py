import matplotlib.pyplot as plt
import numpy as np

# Environmental deterioration
alpha_environment = 1
beta_environment = 1

# Deterioration through traffic
alpha_traffic = 0.5
beta_traffic = 0.3

# Weight of the degradation
weight_environment = 0.35
weight_traffic = 0.65

pci = 100
time_range = range(0, 51)
for age in time_range:
    pci = pci - (weight_environment * np.random.gamma(alpha_environment * age, beta_environment)
                 + weight_traffic * np.random.gamma(alpha_traffic * age, beta_traffic))
    print(pci)
