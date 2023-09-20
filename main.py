# Main program

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from traffic import function_lib as tf
from pavement import function_lib as pv
from network_efficiency import function_lib as ne

road_network_1 = nx.MultiDiGraph()
road_network_1.add_node(1)
road_network_1.add_node(2)
road_network_1.add_edge(1, 2, key=0, length=100, lanes=2, velocity=100, AAT=450, PCI=100, time=60)   # Der key-Parameter hilft dabei, die verschiedenen Kanten zu unterscheiden.
road_network_1.add_edge(2, 1, key=1, length=100, lanes=2, velocity=100, AAT=700, PCI=100, time=60)

# nx.draw(road_network_1)
# plt.show()

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

# Simulation of the network for 100 years
for j in range(100):
    for u, v, key, data in road_network_1.edges(data=True, keys=True):
        data['PCI'] = pv.pavement_deterioration(data['PCI'], PCI_groups, transition_matrix, initial_status, j)
        data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'])
        data['time'] = tf.travel_time(data['velocity'], data['length'])

    # Saving the PCI and velocity values of edge (1, 2, 0) for exemplary
    PCI_history.append(road_network_1[1][2][0]['PCI'])
    velocity_history.append(road_network_1[1][2][0]['velocity'])
    travel_time_history.append(road_network_1[1][2][0]['time'])

import matplotlib.pyplot as plt

# Erstellen Sie eine Figur und Achsen für die Subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot für PCI history
axs[0].plot(range(100), PCI_history, color='tab:red')
axs[0].set_ylabel('PCI')
axs[0].set_title("PCI history")
axs[0].grid(True)

# Plot für Velocity history
axs[1].plot(range(100), velocity_history, color='tab:red')
axs[1].set_ylabel('Velocity [km/h]')
axs[1].set_title("Velocity history")
axs[1].grid(True)

# Plot für Travel Time history
axs[2].plot(range(100), travel_time_history, color='tab:red')
axs[2].set_xlabel('Year')
axs[2].set_ylabel('Travel time [min]')
axs[2].set_title("Travel Time history")
axs[2].grid(True)

# Anzeige der Plots
plt.tight_layout()
plt.show()

# Alle Kanten mit ihren Attributen als Liste ausgeben
road_list_1 = list(road_network_1.edges(data=True, keys=True))
print(road_list_1)

# print(ne.network_efficiency(road_network_1))
