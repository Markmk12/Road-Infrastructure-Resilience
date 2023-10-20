import numpy as np
import pandas as pd
import copy
import networkx as nx
import matplotlib.pyplot as plt
from function_library import system, traffic_dynamics as tf, pavement as pv, maintenance as ma
import itertools
import math
import time

# Notes and TODOs:
# Maintenance
# Spikes in NetEff? Have set a filter for now, Spikes because of wide gamma distribution for larger t ?

# Measure computation time
start = time.time()

# Import a road network (You can find examples in: network_import/networks_of_investigation)
# imported_road_network = nx.read_gexf("network_import/networks_of_investigation/germany_hannover.gexf")

# Perfect state of the road network
# road_network_0 = imported_road_network

# Test Case
road_network_0 = nx.MultiDiGraph()
road_network_0.add_node(1)
road_network_0.add_node(2)
road_network_0.add_node(3)
road_network_0.add_node(4)
road_network_0.add_node(5)
road_network_0.add_edge(1, 2, key=0, highway='primary', length=100000, capacity=100000, lanes=2, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
road_network_0.add_edge(2, 1, key=1, highway='primary', length=100000, capacity=100000, lanes=2, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
road_network_0.add_edge(2, 3, key=2, highway='secondary', length=100000, capacity=15000, lanes=2, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
road_network_0.add_edge(1, 3, key=3, highway='secondary', length=100000, lanes=1, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
road_network_0.add_edge(3, 4, key=4, highway='secondary', length=100000, lanes=1, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
road_network_0.add_edge(2, 4, key=5, highway='primary', length=100000, lanes=2, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
road_network_0.add_edge(4, 2, key=6, highway='primary', length=100000, lanes=2, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
road_network_0.add_edge(4, 5, key=7, highway='primary', length=100000, lanes=1, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
road_network_0.add_edge(5, 4, key=8, highway='primary', length=100000, lanes=1, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)

# Ideal network efficiency (target efficiency)
target_efficiency = system.network_efficiency(road_network_0)

# Road network for simulation
# Import of a graph
road_network_1 = road_network_0

# Randomly sampling PCI and age to each edge and adjust correspond velocity and travel time
# start1 = time.time()
for _, _, key, data in road_network_1.edges(keys=True, data=True):
    data['PCI'] = np.random.choice(list(range(70, 100)))
    data['age'] = 0
    # data['age'] = np.random.choice(list(range(4)))
    data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'], data['maxspeed'])
    data['time'] = tf.travel_time(data['velocity'], data['length'])
# end1 = time.time()
# print("Execution time of randomization: ", str(end1-start1), "[sec]")

# Debugging (show all edges of the graph with their attributes)
# print(road_network_1)
# for u, v, attrs in road_network_1.edges(data=True):
#     print(f"Edge: ({u}, {v}), Attributes: {attrs}")

# Visualize the graph
# pos = nx.spring_layout(road_network_1)
# nx.draw(road_network_1, pos, with_labels=True, node_size=500)
# labels = nx.get_edge_attributes(road_network_1, 'PCI')                    # (doesn't work for MultiDiGraphs)
# nx.draw_networkx_edge_labels(road_network_1, pos, edge_labels=labels)     # (doesn't work for MultiDiGraphs)
# plt.show()

# Simulation time period and sample size
simulation_time_period = range(0, 46)                          # 0-101 years        # 0-601 months = 50 years # 0-46
sample_size = 5                                                 # increase sample size ! 300  # 50 ?

# Quality levels of road maintenance
quality_levels = ["none", "moderate", "extensive"]

# Generate all strategy paths and time points of decision-making
# Generate all tuple for one time point
tuples = list(itertools.product(quality_levels, repeat=2))

# Generate all possible paths for 3 time points (0,15,30 years)
all_strategies = list(itertools.product(tuples, repeat=3))

# Debugging
# print(all_strategies[0])

# Set resilience threshold
res_threshold = 0.5

# Info of inputs before starting the calculation
print(road_network_1)
print("Simulation time period: ", simulation_time_period[0], "-", simulation_time_period[-1], "[Years]")
print("Sample size: " + str(sample_size))
print("Resilience threshold: ", str(res_threshold))

# Results
strategies_matrix_efficiency = np.zeros((len(all_strategies), len(simulation_time_period)))
strategies_matrix_resilience = np.zeros(len(all_strategies))
strategies_matrix_costs = np.zeros((len(all_strategies)))

# Brute-force search
for idx, strategy in enumerate(all_strategies):

    # Matrix
    efficiency_matrix = np.zeros((sample_size, len(simulation_time_period)))
    pci_matrix = np.zeros((sample_size, len(simulation_time_period)))
    costs_matrix = np.zeros((sample_size, len(simulation_time_period)))

    # Simulation of the network efficiency over 100 years
    for sample in range(sample_size):

        # Create a copy of the road network to avoid modifying the original
        temp_network = copy.deepcopy(road_network_1)

        # start2 = time.time()

        # Calculation of the network efficiency
        for t in simulation_time_period:

            # start3 = time.time()

            # Changing the strategy configuration (tuple) every 15 years
            if 0 <= t <= 14:
                quality_level = strategy[0]
            if 15 <= t <= 29:
                quality_level = strategy[1]
            if t >= 30:
                quality_level = strategy[2]

            # Modify the network for time t
            for u, v, key, data in temp_network.edges(keys=True, data=True):

                data['age'] = data['age'] + 1
                data['PCI'] = data['PCI'] - pv.pavement_deterioration_random_process(data['age'])

                # Logical correction (PCI values could only be in an interval 0-100)
                data['PCI'] = max(0, min(data['PCI'], 100))

                # Inspection and Maintenance
                # Inspection
                if data['maintenance'] == 'no':
                    data['maintenance'] = ma.inspection(data['PCI'], data['maintenance'])
                    data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
                    data['time'] = tf.travel_time(data['velocity'], data['length'])

                # Start of measures
                # Start of preventive maintenance
                elif data['maintenance'] == 'preventive_measures_planning':
                    travel_time_impact, duration, *_ = ma.preventive_maintenance(quality_level[0], data['PCI'], data['length'], data['lanes'])

                    data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
                    data['time'] = tf.travel_time(data['velocity'], data['length'])*travel_time_impact

                    data['duration'] = duration
                    if data['duration'] == 0:
                        data['maintenance'] = 'preventive_measures_ending'
                    else:
                        data['maintenance'] = 'preventive_measures_ongoing'

                # Start of corrective maintenance
                elif data['maintenance'] == 'corrective_measures_planning':
                    travel_time_impact, duration, *_ = ma.corrective_maintenance(quality_level[1], data['PCI'], data['length'], data['age'], data['lanes'])

                    data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
                    data['time'] = tf.travel_time(data['velocity'], data['length']) * travel_time_impact

                    data['duration'] = duration
                    if data['duration'] == 0:
                        data['maintenance'] = 'corrective_measures_ending'
                    else:
                        data['maintenance'] = 'corrective_measures_ongoing'

                # Ongoing measures
                # Ongoing of preventive maintenance
                elif data['maintenance'] == 'preventive_measures_ongoing' and data['duration'] != 0:
                    data['duration'] = data['duration']-1
                    if data['duration'] == 0:
                        data['maintenance'] = 'preventive_measures_ending'

                # Ongoing of corrective maintenance
                elif data['maintenance'] == 'corrective_measures_ongoing' and data['duration'] != 0:
                    data['duration'] = data['duration'] - 1
                    if data['duration'] == 0:
                        data['maintenance'] = 'corrective_measures_ending'

                # Completed measures
                # Completed preventive maintenance
                elif data['maintenance'] == 'preventive_measures_ending' and data['duration'] == 0:
                    _, duration, new_pci, age_reset, costs = ma.preventive_maintenance(quality_level[0], data['PCI'], data['length'], data['lanes'])

                    data['age'] = data['age'] - age_reset
                    data['PCI'] = new_pci
                    data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
                    data['time'] = tf.travel_time(data['velocity'], data['length'])
                    data['maintenance'] = 'no'

                    # Costs of the preventive measure
                    costs_matrix[sample, t] = costs_matrix[sample, t] + costs

                # Completed corrective maintenance
                elif data['maintenance'] == 'corrective_measures_ending' and data['duration'] == 0:
                    _, duration, new_pci, age_reset, costs, lanes = ma.corrective_maintenance(
                        quality_level[1], data['PCI'], data['length'], data['age'], data['length'])

                    data['age'] = data['age'] - age_reset
                    data['PCI'] = new_pci
                    data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
                    data['time'] = tf.travel_time(data['velocity'], data['length'])
                    data['maintenance'] = 'no'

                    # Costs of the corrective measure
                    costs_matrix[sample, t] = costs_matrix[sample, t] + costs

                # Debugging
                # print(temp_network[1][2][0]['PCI'])
                # print(temp_network[1][2][0]['maintenance'])

            # Network Efficiency at time t
            efficiency_sample_t = system.network_efficiency(temp_network)
            # Sample Normalizing
            normed_sample_efficiency_t = efficiency_sample_t / target_efficiency
            # Save the normed efficiency at time t in a matrix (rows = sample, columns = time)
            efficiency_matrix[sample, t] = normed_sample_efficiency_t

            # end3 = time.time()
            # print("Execution time of one time step: ", str(end3 - start3), "[sec]")

        # end2 = time.time()
        # print("Execution time of one sample: ", str(end2 - start2), "[sec]")

    # Delete all rows (sample) in the matrix that have a row element greater than 1
    efficiency_matrix = efficiency_matrix[~(efficiency_matrix > 1).any(axis=1)]

    # Calculate the efficiency mean of each column and save it in an extra row
    mean_efficiency_row = efficiency_matrix.mean(axis=0)
    efficiency_matrix = np.vstack([efficiency_matrix, mean_efficiency_row])

    # Calculate the costs mean of each column and save it in an extra row
    mean_costs_row = costs_matrix.mean(axis=0)
    costs_matrix = np.vstack([costs_matrix, mean_costs_row])

    # Resilience
    resilience = system.resilience_metric(efficiency_matrix[-1, :], 1, len(simulation_time_period))
    strategies_matrix_resilience[idx] = resilience

    # Save the estimated efficiency es an entry of strategies_matrix_efficiency
    strategies_matrix_efficiency[idx, :] = mean_efficiency_row

    # Save the estimated costs as an entry of strategies_matrix_costs (These are the expected total costs of the strategy)
    strategies_matrix_costs[idx] = np.sum(mean_costs_row, axis=0)


# Debugging
# print(strategies_matrix_resilience)
# print(strategies_matrix_efficiency)
# print(strategies_matrix_costs)

# Find the best strategy
indices = np.where(strategies_matrix_resilience > res_threshold)
values = strategies_matrix_resilience[indices]
costs = strategies_matrix_costs[indices[0]]

# Debugging
# print(indices)
# print(values)
# print(costs)

# Print of the indices and values and save them in a list
best_strategies_list = []
for idx, value, cost in zip(indices[0], values, costs):
    strategy = f"Strategy: {idx}, Resilience: {value}, Expected total costs: {cost}"
    best_strategies_list.append(strategy)

# print(best_strategies_list)

# Choosing the best strategy based on resilience
sorted_strategies = sorted(best_strategies_list, key=lambda x: float(x.split("Expected total costs: ")[1]))

# Der Eintrag mit den kleinsten "Expected total costs" ist nun der erste in der sortierten Liste:
best_strategy = sorted_strategies[0]

print(sorted_strategies)
print(best_strategy)

# Get index of the best strategy as an integer
strategy_str = best_strategy.split(",")[0]  # Dies teilt den String bei jedem Komma und nimmt den ersten Teil, also "Strategy: 5"
idx_str = strategy_str.split(":")[1].strip()  # Dies teilt den String bei ':' und nimmt den zweiten Teil, also " 5", und entfernt dann die Leerzeichen
idx_best = int(idx_str)

# Path
print(all_strategies[idx_best])

# Plot
# efficiency_mean_best_strategy = strategies_matrix_efficiency[-1, :]
plt.step(simulation_time_period, strategies_matrix_efficiency[idx_best, :], color='red', linestyle='-')

plt.xlabel('Simulation Time Period [Year]')
plt.ylabel('Network Efficiency [-]')
plt.title('Network Efficiency')
plt.grid(True)
plt.grid(which='major', color='#DDDDDD', linewidth=0.9)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9)
plt.minorticks_on()
plt.ylim(0, 1)
plt.show()

# Plot of all strategies
# for row in strategies_matrix_efficiency:
#     plt.step(simulation_time_period, row, color='lightgray')
# plt.show()

# Plot of the efficiency for the best resilient strategies
number_of_plots = len(indices[0])
if number_of_plots <= 0:
    print("Number of plots cannot be zero or negative!")
else:
    fig, axes = plt.subplots(number_of_plots, 1, figsize=(8, 4*number_of_plots))  # 4*number_of_plots gives each plot enough vertical space.

    for idx, row_index in enumerate(indices[0]):
        ax = axes[idx]  # Select the current subplot.
        ax.step(simulation_time_period, strategies_matrix_efficiency[row_index], color='red', linestyle='-')
        ax.set_xlabel('Time [Year]')
        ax.set_ylabel('Network Efficiency [-]')
        ax.set_title(f'Network Efficiency for Index {row_index}')
        ax.grid(True)
        ax.grid(which='major', color='#DDDDDD', linewidth=0.9)
        ax.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9)
        ax.minorticks_on()

    plt.tight_layout()  # Provides enough space between the subplots.
    plt.show()
