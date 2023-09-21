# Main program

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from traffic import function_lib as tf
from pavement import function_lib as pv
from network_efficiency import function_lib as eff

road_network_1 = nx.Graph()
road_network_1.add_node(1)
road_network_1.add_node(2)
road_network_1.add_node(3)
road_network_1.add_node(4)
road_network_1.add_node(5)
road_network_1.add_edge(1, 2, key=0, length=100, lanes=4, velocity=100, AAT=450, PCI=100, time=60)   # Der key-Parameter hilft dabei, die verschiedenen Kanten zu unterscheiden.
road_network_1.add_edge(2, 1, key=1, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)
road_network_1.add_edge(2, 3, key=2, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)
road_network_1.add_edge(1, 3, key=3, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)
road_network_1.add_edge(3, 4, key=4, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)
road_network_1.add_edge(2, 4, key=5, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)
road_network_1.add_edge(4, 5, key=6, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)

# Plot graph
nx.draw(road_network_1)
plt.show()

# Alle Kanten mit ihren Attributen als Liste ausgeben
# road_list_1 = list(road_network_1.edges(data=True, keys=True))
# print(road_list_1)

# Define PCI groups
PCI_groups = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0])
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


PCI_history = []
velocity_history = []
travel_time_history = []
efficiency_history = []

# Simulation of the network for 100 years
for j in range(100):
    for u, v, data in road_network_1.edges(data=True):
        data['PCI'] = pv.pavement_deterioration(data['PCI'], PCI_groups, transition_matrix, initial_status, j)
        data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'])
        data['time'] = tf.travel_time(data['velocity'], data['length'])

    # Saving the PCI and velocity values of edge (1, 2, 0) for exemplary
    PCI_history.append(road_network_1[1][2]['PCI'])
    velocity_history.append(road_network_1[1][2]['velocity'])
    travel_time_history.append(road_network_1[1][2]['time'])

    # Network Efficiency
    efficiency_t = eff.network_efficiency(road_network_1)

    # Saving network efficiency history data
    efficiency_history.append(efficiency_t)

# Create a figure and axes for the subplots in a 2x2 layout
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Plot for PCI history for edge (1, 2)
axs[0, 0].plot(range(100), PCI_history, color='tab:red')
axs[0, 0].set_ylabel('PCI')
axs[0, 0].set_title("PCI prediction for edge (1, 2)")
axs[0, 0].grid(True)

# Plot for Velocity history for edge (1, 2)
axs[0, 1].plot(range(100), velocity_history, color='tab:red')
axs[0, 1].set_ylabel('Velocity [km/h]')
axs[0, 1].set_title("Velocity prediction for edge (1, 2)")
axs[0, 1].grid(True)

# Plot for Travel Time history for edge (1, 2)
axs[1, 0].plot(range(100), travel_time_history, color='tab:red')
axs[1, 0].set_xlabel('Year')
axs[1, 0].set_ylabel('Travel time [min]')
axs[1, 0].set_title("Travel time prediction for edge (1, 2)")
axs[1, 0].grid(True)

# Plot for Efficiency History
axs[1, 1].plot(range(100), efficiency_history, color='tab:red')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Network Efficiency [-]')
axs[1, 1].set_title("Network efficiency prediction")
axs[1, 1].grid(True)

# Display the plots
plt.tight_layout()
plt.show()


# Alle Kanten mit ihren Attributen als Liste ausgeben
road_list_1 = list(road_network_1.edges(data=True))
print(road_list_1)

efficiency = eff.network_efficiency(road_network_1)
print(efficiency)
