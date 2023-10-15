import numpy as np
import pandas as pd
import copy
import networkx as nx
import matplotlib.pyplot as plt
from function_library import system, traffic_dynamics as tf, pavement as pv, maintenance as ma
import time

# Notes and TODOs:
# Maintenance
# Spikes in NetEff? Have set a filter for now, Spikes because of wide gamma distribution for larger t ?

# Measure computation time
start = time.time()

# Import a road network (You can find examples in: network_import/networks_of_investigation)
# imported_road_network = nx.read_gexf("network_import/networks_of_investigation/germany_bennigsen.gexf")

# Perfect state of the road network
# road_network_0 = imported_road_network

# Test Case
road_network_0 = nx.MultiDiGraph()
road_network_0.add_node(1)
road_network_0.add_node(2)
road_network_0.add_node(3)
road_network_0.add_node(4)
road_network_0.add_node(5)
road_network_0.add_edge(1, 2, key=0, highway='primary', length=100000, capacity=100000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)
road_network_0.add_edge(2, 1, key=1, highway='primary', length=100000, capacity=100000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)
road_network_0.add_edge(2, 3, key=2, highway='secondary', length=100000, capacity=15000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)
road_network_0.add_edge(1, 3, key=3, highway='secondary', length=100000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)
road_network_0.add_edge(3, 4, key=4, highway='secondary', length=100000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)
road_network_0.add_edge(2, 4, key=5, highway='primary', length=100000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)
road_network_0.add_edge(4, 2, key=6, highway='primary', length=100000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)
road_network_0.add_edge(4, 5, key=7, highway='primary', length=100000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)
road_network_0.add_edge(5, 4, key=8, highway='primary', length=100000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)

# Ideal network efficiency (target efficiency)
target_efficiency = system.network_efficiency(road_network_0)

# Road network for simulation
# Import of a graph
road_network_1 = road_network_0

# Randomly sampling PCI and age to each edge and adjust correspond velocity and travel time
# start1 = time.time()
for _, _, key, data in road_network_1.edges(keys=True, data=True):
    data['PCI'] = np.random.choice(list(range(70, 100)))
    # data['age'] = 0
    data['age'] = np.random.choice(list(range(8)))

    # Assign traffic_load based on classification
    # if data['highway'] == 'primary':
    #     data['traffic_load'] = int(np.random.randn() * 30 + 100)  # Standard deviation of 15, mean of 100
    # else:
    #     data['traffic_load'] = int(np.random.randn() * 15 + 50)  # Standard deviation of 10, mean of 50

    data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'], data['maxspeed'])
    data['time'] = tf.travel_time(data['velocity'], data['length'])
# end1 = time.time()
# print("Execution time of randomization: ", str(end1-start1), "[sec]")

# Debugging (show all edges of the graph with their attributes)
print(road_network_1)
for u, v, attrs in road_network_1.edges(data=True):
    print(f"Edge: ({u}, {v}), Attributes: {attrs}")

# Visualize the graph
# pos = nx.spring_layout(road_network_1)
# nx.draw(road_network_1, pos, with_labels=True, node_size=500)
# labels = nx.get_edge_attributes(road_network_1, 'PCI')                    # (doesn't work for MultiDiGraphs)
# nx.draw_networkx_edge_labels(road_network_1, pos, edge_labels=labels)     # (doesn't work for MultiDiGraphs)
# plt.show()

# Simulation time period and sample size
simulation_time_period = range(0, 101)                          # 0-101 years        # 0-601 months = 50 years
sample_size = 5                                                 # increase sample size ! 300  # 50 ?

# Quality levels of road maintenance
quality_levels = ["none", "moderate", "extensive"]
quality_level = 'moderate'

# Generate all strategy paths and time points of decision-making


# Set resilience threshold
res_threshold = 0.8

# Info of inputs before starting the calculation
print(road_network_1)
print("Simulation time period: ", simulation_time_period[0], "-", simulation_time_period[-1], "[Years]")
print("Sample size: " + str(sample_size))
print("Resilience threshold: ", str(res_threshold))

# Matrix
efficiency_matrix = np.zeros((sample_size, len(simulation_time_period)))
pci_matrix = np.zeros((sample_size, len(simulation_time_period)))

# Simulation of the network efficiency over 100 years
for sample in range(sample_size):

    # Create a copy of the road network to avoid modifying the original
    temp_network = copy.deepcopy(road_network_1)

    # start2 = time.time()

    # Simulation of the network efficiency for each sample
    for t in simulation_time_period:

        # start3 = time.time()

        # Modify the network for time t
        for u, v, key, data in temp_network.edges(keys=True, data=True):

            data['age'] = data['age'] + 1
            data['PCI'] = data['PCI'] - pv.pavement_deterioration_random_process(data['age'])

            # Logical correction
            if data['PCI'] <= 0:
                data['PCI'] = 0
            elif data['PCI'] > 100:
                data['PCI'] = 100

            # Inspection and Maintenance
            # Inspection
            if data['maintenance'] == 'no':
                data['maintenance'] = ma.inspection(data['PCI'], data['maintenance'])
                data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
                data['time'] = tf.travel_time(data['velocity'], data['length'])

            # Ongoing measures
            # Ongoing preventive maintenance
            elif data['maintenance'] == 'preventive_measures_planning_and_realization':
                travel_time_impact, *_ = ma.preventive_maintenance(quality_level, data['PCI'], data['length'])

                data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
                data['time'] = tf.travel_time(data['velocity'], data['length'])*travel_time_impact
                data['maintenance'] = 'preventive_measures_ongoing'

            # Ongoing corrective maintenance
            elif data['maintenance'] == 'corrective_measures_planning_and_realization':
                travel_time_impact, *_ = ma.corrective_maintenance(quality_level, data['PCI'], data['length'], data['age'])

                data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
                data['time'] = tf.travel_time(data['velocity'], data['length']) * travel_time_impact
                data['maintenance'] = 'corrective_measures_ongoing'

            # Completed measures
            # Completed preventive maintenance
            elif data['maintenance'] == 'preventive_measures_ongoing':
                _, duration, new_pci, maintenance_status, age_reset, costs = ma.preventive_maintenance(quality_level, data['PCI'], data['length'])

                data['age'] = data['age'] - age_reset
                data['PCI'] = new_pci
                data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
                data['time'] = tf.travel_time(data['velocity'], data['length'])
                data['maintenance'] = maintenance_status

            # Completed corrective maintenance
            elif data['maintenance'] == 'corrective_measures_ongoing':
                _, duration, new_pci, maintenance_status, age_reset, costs = ma.corrective_maintenance(
                    quality_level, data['PCI'], data['length'], data['age'])

                data['age'] = data['age'] - age_reset
                data['PCI'] = new_pci
                data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
                data['time'] = tf.travel_time(data['velocity'], data['length'])
                data['maintenance'] = maintenance_status


            # elif data['maintenance'] == 'scheduled':
            #     data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
            #     data['time'] = tf.travel_time(data['velocity'], data['length'])*1.25        # increase travel time x1.25
            #     data['age'] = data['age'] + 1
            #     data['maintenance'] = 'yes'
            #
            # elif data['maintenance'] == 'yes':
            #     data['PCI'] = ma.simple_maintenance(data['PCI'])
            #     data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
            #     data['time'] = tf.travel_time(data['velocity'], data['length'])
            #     data['age'] = 0
            #     data['maintenance'] = 'no'

            # data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
            # data['time'] = tf.travel_time(data['velocity'], data['length'])

            # Debugging
            # print(temp_network[1][2][0]['PCI'])
            # print(temp_network[1][2][0]['maintenance'])

        # Sample Network Efficiency at time t
        efficiency_sample_t = system.network_efficiency(temp_network)
        # Sample Normalizing
        normed_sample_efficiency_t = efficiency_sample_t / target_efficiency
        # Save the normed efficiency at time t in a matrix (rows = sample, columns = time)
        efficiency_matrix[sample, t] = normed_sample_efficiency_t

        # Save PCI value of edge
        pci_matrix[sample, t] = temp_network[1][2][0]['PCI']

        # end3 = time.time()
        # print("Execution time of one time step: ", str(end3 - start3), "[sec]")

    # end2 = time.time()
    # print("Execution time of one sample: ", str(end2 - start2), "[sec]")

# Delete all rows (sample) in the matrix that have a row element greater than 1
efficiency_matrix = efficiency_matrix[~(efficiency_matrix > 1).any(axis=1)]

# Calculate the efficiency mean of each column and save it in an extra row
mean_efficiency_row = efficiency_matrix.mean(axis=0)
efficiency_matrix = np.vstack([efficiency_matrix, mean_efficiency_row])

# Debugging
# print(efficiency_matrix)

# Resilience
resilience = system.resilience_metric(efficiency_matrix[-1, :], 1, len(simulation_time_period))

# Print of the results
# print("The predicted normalized Network Efficiency is: " + str(normed_efficiency_history[-1]))
print("Resilience: ", str(resilience))

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

# Plot of the samples PCI of first edge
for row in pci_matrix[:-1]:
    plt.step(simulation_time_period, row, color='lightgray')

plt.xlabel('Simulation Time Period [Year]')
plt.ylabel('PCI [-]')
plt.title('PCI of first edge')
plt.grid(True)
plt.grid(which='major', color='#DDDDDD', linewidth=0.9)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9)
plt.minorticks_on()
plt.show()
