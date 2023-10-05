import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from function_library import traffic_dynamics as tf, pavement as pv
from network_efficiency import function_lib as eff

# Notes and TODOs:
# The sampling should not update the graph of time step t
# Note 1
# ...

# Ideal road network
road_network_0 = nx.Graph()
road_network_0.add_node(1)
road_network_0.add_node(2)
road_network_0.add_node(3)
road_network_0.add_node(4)
road_network_0.add_node(5)
road_network_0.add_edge(1, 2, key=0, length=100, lanes=4, velocity=100, AAT=450, PCI=100, time=60, maintenance=0)   # Der key-Parameter hilft dabei, die verschiedenen Kanten zu unterscheiden.
road_network_0.add_edge(2, 1, key=1, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)
road_network_0.add_edge(2, 3, key=2, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)
road_network_0.add_edge(1, 3, key=3, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)
road_network_0.add_edge(3, 4, key=4, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)
road_network_0.add_edge(2, 4, key=5, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)
road_network_0.add_edge(4, 5, key=6, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)

# Road network for simulation
road_network_1 = nx.Graph()
road_network_1.add_node(1)
road_network_1.add_node(2)
road_network_1.add_node(3)
road_network_1.add_node(4)
road_network_1.add_node(5)
road_network_1.add_edge(1, 2, key=0, length=100, lanes=4, velocity=100, AAT=450, PCI=100, time=60, maintenance=0)   # Der key-Parameter hilft dabei, die verschiedenen Kanten zu unterscheiden.
road_network_1.add_edge(2, 1, key=1, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)
road_network_1.add_edge(2, 3, key=2, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)
road_network_1.add_edge(1, 3, key=3, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)
road_network_1.add_edge(3, 4, key=4, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)
road_network_1.add_edge(2, 4, key=5, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)
road_network_1.add_edge(4, 5, key=6, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60, maintenance=0)

# Plot graph
nx.draw(road_network_1)
plt.show()


# All edges with attributes as list
# road_list_1 = list(road_network_1.edges(data=True, keys=True))
# print(road_list_1)


# Define PCI groups
PCI_groups = np.array([100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 0])

# Lists
PCI_history = []
velocity_history = []
travel_time_history = []
normed_efficiency_history = []

# Lists
normed_efficiency_t_samples = []

# Simulation time period
simulation_time_period = range(1, 10)

# Simulation of the network efficiency over 100 years
for t in simulation_time_period:
    for _ in range(300):
        for u, v, data in road_network_1.edges(data=True):

            # Simulation for every edge
            data['PCI'] = pv.pavement_deterioration_gamma_process(data['PCI'], t)
            data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'])
            data['time'] = tf.travel_time(data['velocity'], data['length'])

        # Saving the results for each sample
        # PCI

        # Network Efficiency
        efficiency_t_sample = eff.network_efficiency(road_network_1)
        normed_efficiency_t = efficiency_t_sample / eff.network_efficiency(road_network_0)
        normed_efficiency_t_samples.append(normed_efficiency_t)

    # Means of the samples at time t
    efficiency_t_mean = np.mean(normed_efficiency_t_samples)

    # Saving the means
    normed_efficiency_history.append(efficiency_t_mean)


# Simulation of the network for 100 years
for t in simulation_time_period:
    for u, v, data in road_network_1.edges(data=True):
        #data['PCI'] = pv.pavement_deterioration_markov_chain(data['PCI'], PCI_groups, transition_matrix, initial_status,
        #                                                     t)
        data['PCI'] = pv.pavement_deterioration_gamma_process(data['PCI'], t)
        data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'])
        data['time'] = tf.travel_time(data['velocity'], data['length'])

    # Saving the PCI and velocity values of edge (1, 2, 0) for exemplary
    PCI_history.append(road_network_1[1][2]['PCI'])
    velocity_history.append(road_network_1[1][2]['velocity'])
    travel_time_history.append(road_network_1[1][2]['time'])

    # Normed network Efficiency at time t
    efficiency_t = eff.network_efficiency(road_network_1)
    normed_efficiency_t = efficiency_t / eff.network_efficiency(road_network_0)

    # Saving network efficiency prediction data
    normed_efficiency_history.append(normed_efficiency_t)

# Create a figure and axes for the subplots in a 2x2 layout
fig, axs = plt.subplots(2, 2, figsize=(12, 12))

# Plot for PCI prediction for edge (1, 2)
axs[0, 0].plot(simulation_time_period, PCI_history, color='tab:red')
axs[0, 0].set_ylabel('PCI')
axs[0, 0].set_title("PCI prediction for edge (1, 2)")
axs[0, 0].grid(True)

# Plot for Velocity prediction for edge (1, 2)
axs[0, 1].plot(simulation_time_period, velocity_history, color='tab:red')
axs[0, 1].set_ylabel('Velocity [km/h]')
axs[0, 1].set_title("Velocity prediction for edge (1, 2)")
axs[0, 1].grid(True)

# Plot for Travel Time prediction for edge (1, 2)
axs[1, 0].plot(simulation_time_period, travel_time_history, color='tab:red')
axs[1, 0].set_xlabel('Year')
axs[1, 0].set_ylabel('Travel time [min]')
axs[1, 0].set_title("Travel time prediction for edge (1, 2)")
axs[1, 0].grid(True)

# Plot for efficiency prediction
axs[1, 1].plot(simulation_time_period, normed_efficiency_history, color='tab:red')
axs[1, 1].set_xlabel('Year')
axs[1, 1].set_ylabel('Normed network Efficiency [-]')
axs[1, 1].set_title("Normed network efficiency prediction")
axs[1, 1].grid(True)

# Display the plots
plt.tight_layout()
plt.show()


# Alle Kanten mit ihren Attributen als Liste ausgeben
road_list_1 = list(road_network_1.edges(data=True))
print(road_list_1)

# Network efficiency
efficiency = eff.network_efficiency(road_network_1)
print(efficiency)

# Normalized efficiency
norm_efficiency = efficiency / eff.network_efficiency(road_network_0)
print("The Normalized Network Efficiency is: " + str(norm_efficiency))
