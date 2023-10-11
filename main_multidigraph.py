import numpy as np
import pandas as pd
import copy
import networkx as nx
import matplotlib.pyplot as plt
from function_library import system, traffic_dynamics as tf, pavement as pv
import time

# Notes and TODOs:
# Maintenance
# Ausreißer???  -> Erstmal ein Filter

# Measure computation time
start = time.time()

# Import Road Network
imported_road_network = nx.read_gexf("network_import/networks_of_investigation/germany_bennigsen.gexf")

# Ideal road network
road_network_0 = imported_road_network

# Test Case
# road_network_0 = nx.MultiDiGraph()
# road_network_0.add_node(1)
# road_network_0.add_node(2)
# road_network_0.add_node(3)
# road_network_0.add_node(4)
# road_network_0.add_node(5)
# road_network_0.add_edge(1, 2, key=0, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=450, PCI=100, time=60, maintenance=0, age=0)
# road_network_0.add_edge(2, 1, key=1, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_0.add_edge(2, 3, key=2, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_0.add_edge(1, 3, key=3, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_0.add_edge(3, 4, key=4, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_0.add_edge(2, 4, key=5, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_0.add_edge(4, 2, key=6, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_0.add_edge(4, 5, key=7, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_0.add_edge(5, 4, key=8, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)

# Ideal network efficiency (target efficiency)
target_efficiency = system.network_efficiency(road_network_0)


# Road network for simulation
# Import of a graph
road_network_1 = imported_road_network

# Test Case
# road_network_1 = nx.MultiDiGraph()
# road_network_1.add_node(1)
# road_network_1.add_node(2)
# road_network_1.add_node(3)
# road_network_1.add_node(4)
# road_network_1.add_node(5)
# road_network_1.add_edge(1, 2, key=0, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=450, PCI=100, time=60, maintenance=0, age=0)
# road_network_1.add_edge(2, 1, key=1, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_1.add_edge(2, 3, key=2, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_1.add_edge(1, 3, key=3, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_1.add_edge(3, 4, key=4, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_1.add_edge(2, 4, key=5, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_1.add_edge(4, 2, key=6, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_1.add_edge(4, 5, key=7, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
# road_network_1.add_edge(5, 4, key=8, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)

# Randomly sampling PCI and age to each edge and adjust correspond velocity and travel time
for _, _, key, data in road_network_1.edges(keys=True, data=True):
    data['PCI'] = np.random.choice(list(range(70, 100)))
    data['age'] = np.random.choice(list(range(8)))
    data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'], data['maxspeed'])
    data['time'] = tf.travel_time(data['velocity'], data['length'])

# Debugging
# print(road_network_1)
# for u, v, attrs in road_network_1.edges(data=True):
#     print(f"Edge: ({u}, {v}), Attributes: {attrs}")

# Visualize the graph
# pos = nx.spring_layout(road_network_1)
# nx.draw(road_network_1, pos, with_labels=True, node_size=500)
# labels = nx.get_edge_attributes(road_network_1, 'PCI')                    # (doesn't work for MultiDiGraphs)
# nx.draw_networkx_edge_labels(road_network_1, pos, edge_labels=labels)     # (doesn't work for MultiDiGraphs)
# plt.show()

# Lists
normed_efficiency_t_samples = []
normed_efficiency_history = []
pci_mean_history = []
mean_efficiency_history = []

# Simulation time period and sample size
simulation_time_period = range(0, 101)                          # 0-101 years        # 0-601 months = 50 years
sample_size = 5                                                 # increase sample size ! 300  # 50 ?

# Info of inputs before starting the calculation
print(imported_road_network)
print("Simulation time period: ", simulation_time_period[0], "-", simulation_time_period[-1], "[Years]")
print("Sample size: " + str(sample_size), "[-]")

# Matrix
efficiency_matrix = np.zeros((sample_size, len(simulation_time_period)))

# Simulation of the network efficiency over 100 years
for sample in range(sample_size):

    # Simulation of the network efficiency for each sample
    for t in simulation_time_period:

        # Create a copy of the road network to avoid modifying the original
        temp_network = copy.deepcopy(road_network_1)

        # Modify the network for time t
        for u, v, key, data in temp_network.edges(keys=True, data=True):

            data['PCI'] = data['PCI'] - pv.pavement_deterioration_gamma_process_alternative(data['PCI'], t)

            if data['PCI'] <= 0:
                data['PCI'] = 0
            elif data['PCI'] > 100:
                data['PCI'] = 100

            data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
            data['time'] = tf.travel_time(data['velocity'], data['length'])
            data['age'] = data['age'] + 1

        # Sample Network Efficiency at time t
        efficiency_sample_t = system.network_efficiency(temp_network)
        # Sample Normalizing
        normed_sample_efficiency_t = efficiency_sample_t / target_efficiency
        # Save the normed efficiency at time t in a matrix (rows = sample, columns = time)
        efficiency_matrix[sample, t] = normed_sample_efficiency_t

# Delete all rows (sample) in the matrix that have a row element greater than 1
efficiency_matrix = efficiency_matrix[~(efficiency_matrix > 1).any(axis=1)]

# Calculate the efficiency mean of each column and save it in an extra row
mean_efficiency_row = efficiency_matrix.mean(axis=0)
efficiency_matrix = np.vstack([efficiency_matrix, mean_efficiency_row])

# Debugging
# print(efficiency_matrix)

# Resilience
# resilience = system.resilience_metric(normed_efficiency_history, 1, len(simulation_time_period))        # korrigieren!!!!
resilience = system.resilience_metric(efficiency_matrix[-1, :], 1, len(simulation_time_period))

# Print of the results
# print("The predicted normalized Network Efficiency is: " + str(normed_efficiency_history[-1]))
print("Resilience: ", str(resilience), "[-]")

# Measure computation time
end = time.time()
print("Execution time: ", str(end-start), "[sec]")

# Plot of the samples
for row in efficiency_matrix[:-1]:
    plt.step(simulation_time_period, row, color='lightgray')

# Plot of the means
mean_values = efficiency_matrix[-1, :]
plt.step(simulation_time_period, mean_values, color='red', linestyle='-')

plt.xlabel('Simulation Time Period [Year]')
plt.ylabel('Network Efficiency [-]')
plt.title('Network Efficiency')
plt.grid(True)
plt.grid(which='major', color='#DDDDDD', linewidth=0.9)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9)
plt.minorticks_on()
plt.show()
