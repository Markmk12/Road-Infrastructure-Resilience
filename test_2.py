import numpy as np
from pavement import function_lib as pv

# Define PCI groups
pci_groups = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0])
# PCI_groups = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])


# Define a Probability Transition Matrix and initial vector
transition_matrix = np.array([
    [0.95, 0.05, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0.85, 0.15, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0.7, 0.3, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0.65, 0.35, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0.6, 0.4, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0.5, 0.5, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0.4, 0.6, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0.3, 0.7, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.15, 0.85],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
])

initial_status = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

p = initial_status.dot(np.linalg.matrix_power(transition_matrix, 30))

# MC simulation
sample = np.random.choice(pci_groups, 300, p=p)

# Expected value (arithmetic mean)
mean = np.mean(sample)

print(p)
print(sample)
print(mean)
