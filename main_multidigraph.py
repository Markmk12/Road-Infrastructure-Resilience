import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
from function_library import system, traffic_dynamics as tf, pavement as pv, maintenance as ma
import itertools
import time
import os
import sys

# Notes and TODOs:
# nothing

# START HERE: Name the file under which the results will be saved (the results will be stored in the results' folder)
file = 'test_dusseldorf_3'

path = os.path.join('results', file)
if os.path.exists(path):
    print("File already exists. Please change the name or delete the existing file.")
    sys.exit()
else:
    os.makedirs(path)

# Simulation time period and sample size
simulation_time_period = range(0, 31)                          # 0-101 years        # 0-601 months = 50 years # 0-46
sample_size = 5                                                 # increase sample size ! 300  # 50 ?

# Set resilience threshold
res_threshold = 0.80


# Measure computation time
start = time.time()

# Import a road network (You can find examples in: network_import/networks_of_investigation)
imported_road_network = nx.read_gexf("network_import/networks_of_investigation/graph_51.2277_6.7735.gexf")
# imported_road_network = nx.read_gexf("network_import/networks_of_investigation/simple_test_graphs/simple_test_graph_1.gexf")

# Perfect state of the road network
road_network_0 = imported_road_network
print(road_network_0)

# Ideal network efficiency (optimal efficiency)
optimal_efficiency = system.network_efficiency(road_network_0)
print(optimal_efficiency)

# Road network for simulation
# Import of a graph
road_network_1 = road_network_0

# Randomly sampling PCI and age to each edge and adjust correspond velocity and travel time
# The value of the PCI depends on the age
# start1 = time.time()

for _, _, key, data in road_network_1.edges(keys=True, data=True):

    data['age'] = np.random.choice(list(range(14)))

    if 0 <= data['age'] <= 5:
        data['PCI'] = np.random.choice(list(range(90, 100)))

    elif 6 <= data['age'] <= 11:
        data['PCI'] = np.random.choice(list(range(75, 90)))

    elif 12 <= data['age'] <= 17:
        data['PCI'] = np.random.choice(list(range(50, 75)))

    elif 18 <= data['age'] <= 23:
        data['PCI'] = np.random.choice(list(range(0, 50)))

    data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
    data['time'] = tf.travel_time(data['velocity'], data['length'])

# end1 = time.time()
# print("Execution time of randomization: ", str(end1-start1), "[sec]")

# Debugging (show all edges of the graph with their attributes)
# print(road_network_1)
# for u, v, attrs in road_network_1.edges(data=True):
#     print(f"Edge: ({u}, {v}), Attributes: {attrs}")

# Saving the initial road network
name = "initial_road_network.gexf"
nx.write_gexf(road_network_1, os.path.join(path, name))

# Quality levels of road maintenance
quality_levels = ['sparse', 'moderate', 'extensive']

# Generate all strategy paths and time points of decision-making
# Generate all tuple for one time point
tuples = list(itertools.product(quality_levels, repeat=2))

# Generate all possible paths for 3 time points (0,15,30 years)
all_strategies = list(itertools.product(tuples, repeat=3))

# Debugging
# print(all_strategies[0])

# Info of inputs before starting the calculation
print(road_network_1)
print("Simulation time period: ", simulation_time_period[0], "-", simulation_time_period[-1], "[Years]")
print("Sample size: " + str(sample_size))
print("Resilience threshold: ", str(res_threshold))

# Saving the input parameters
np.save(os.path.join('results', file, 'simulation_time_period.npy'), simulation_time_period)
np.save(os.path.join('results', file, 'sample_size.npy'), sample_size)
np.save(os.path.join('results', file, 'all_strategies.npy'), all_strategies)
np.save(os.path.join('results', file, 'res_threshold.npy'), res_threshold)

# Results
strategies_matrix_efficiency = np.zeros((len(all_strategies), len(simulation_time_period)))
strategies_matrix_resilience = np.zeros(len(all_strategies))
costs_history_matrix = np.zeros((len(all_strategies), len(simulation_time_period)))
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

        start2 = time.time()

        # Calculation of the network efficiency
        for t in simulation_time_period:

            # start3 = time.time()

            # Changing the strategy configuration (tuple) every 10 years
            if 0 <= t <= 9:
                quality_level = strategy[0]
            if 10 <= t <= 19:
                quality_level = strategy[1]
            if t >= 20:
                quality_level = strategy[2]

            # Modify the network for time t
            for u, v, key, data in temp_network.edges(keys=True, data=True):

                # Logical correction (PCI values could only be in an interval 0-100)
                data['PCI'] = max(0, min(data['PCI'], 100))

                # Inspection and Maintenance
                # Inspection
                if data['maintenance'] == 'no':

                    # Deterioration
                    data['age'] = data['age'] + 1
                    data['PCI'] = data['PCI'] - pv.pavement_deterioration_random_process(data['age'])

                    data['maintenance'] = ma.inspection(data['PCI'], data['maintenance'])
                    data['velocity'] = tf.velocity_change_linear(data['PCI'], data['velocity'], data['maxspeed'])
                    data['time'] = tf.travel_time(data['velocity'], data['length'])

                # Start of measures
                # Start of preventive maintenance
                elif data['maintenance'] == 'preventive_measures_planning':

                    # Deterioration
                    data['age'] = data['age'] + 1
                    data['PCI'] = data['PCI'] - pv.pavement_deterioration_random_process(data['age'])

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

                    # Deterioration
                    data['age'] = data['age'] + 1
                    data['PCI'] = data['PCI'] - pv.pavement_deterioration_random_process(data['age'])

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

                    # # No Deterioration?
                    # data['age'] = data['age'] + 1
                    # data['PCI'] = data['PCI'] - pv.pavement_deterioration_random_process(data['age'])

                    data['duration'] = data['duration']-1
                    if data['duration'] == 0:
                        data['maintenance'] = 'preventive_measures_ending'

                # Ongoing of corrective maintenance
                elif data['maintenance'] == 'corrective_measures_ongoing' and data['duration'] != 0:

                    # # No Deterioration?
                    # data['age'] = data['age'] + 1
                    # data['PCI'] = data['PCI'] - pv.pavement_deterioration_random_process(data['age'])

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

            # Network Efficiency at time t
            efficiency_sample_t = system.network_efficiency(temp_network)
            # Sample Normalizing
            normed_sample_efficiency_t = efficiency_sample_t / optimal_efficiency
            # Save the normed efficiency at time t in a matrix (rows = sample, columns = time)
            efficiency_matrix[sample, t] = normed_sample_efficiency_t

            # end3 = time.time()
            # print("Execution time of one time step: ", str(end3 - start3), "[sec]")

        end2 = time.time()
        print("Execution time of one sample: ", str(end2 - start2), "[sec]")

    # Delete all rows (sample) in the matrix that have a row element greater than 1
    # efficiency_matrix = efficiency_matrix[~(efficiency_matrix > 1).any(axis=1)]

    # Calculate the efficiency mean of each column and save it in an extra row
    mean_efficiency_row = efficiency_matrix.mean(axis=0)
    efficiency_matrix = np.vstack([efficiency_matrix, mean_efficiency_row])

    # Calculate the costs mean of each column and save it in an extra row
    mean_costs_row = costs_matrix.mean(axis=0)
    costs_matrix = np.vstack([costs_matrix, mean_costs_row])
    costs_history_matrix[idx] = mean_costs_row

    # Resilience
    resilience = system.resilience_metric(efficiency_matrix[-1, :], len(simulation_time_period))
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

# Choosing the best strategy based on resilience
sorted_best_strategies = sorted(best_strategies_list, key=lambda x: float(x.split("Expected total costs: ")[1]))

# The entry with the smallest "Expected total costs" is now at the top of the sorted list:
best_strategy = sorted_best_strategies[0]

print(sorted_best_strategies)
print(best_strategy)

# Get index of the best strategy as an integer
strategy_str = best_strategy.split(",")[0]
idx_str = strategy_str.split(":")[1].strip()
idx_best = int(idx_str)

# Path configuration of the best strategy
print(all_strategies[idx_best])

# Costs and cumulated costs history of the best strategy path
best_strategy_costs = costs_history_matrix[idx_best]
best_strategy_costs_cumulated = np.cumsum(best_strategy_costs)
# print(best_strategy_costs)
# print(best_strategy_costs_cumulated)
# print(np.sum(best_strategy_costs))

# Saving the results
np.save(os.path.join('results', file, 'strategies_matrix_efficiency.npy'), strategies_matrix_efficiency)
np.save(os.path.join('results', file, 'strategies_matrix_resilience.npy'), strategies_matrix_resilience)
np.save(os.path.join('results', file, 'strategies_matrix_costs.npy'), strategies_matrix_costs)
np.save(os.path.join('results', file, 'sorted_best_strategies.npy'), sorted_best_strategies)
np.save(os.path.join('results', file, 'best_strategy.npy'), best_strategy)
np.save(os.path.join('results', file, 'idx_best.npy'), idx_best)
np.save(os.path.join('results', file, 'best_strategy_costs.npy'), best_strategy_costs)


# Plot of the best strategy
plt.step(simulation_time_period, strategies_matrix_efficiency[idx_best, :], color='red', linestyle='-')

plt.xlabel('Time')
plt.ylabel('Network Efficiency [-]')
plt.title('Network Efficiency')
plt.grid(True)
plt.grid(which='major', color='#DDDDDD', linewidth=0.9)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9)
plt.minorticks_on()
plt.ylim(0, 1)

# Saving the plots in different file formats
plt.savefig(os.path.join('results', file, "efficiency.png"))
plt.savefig(os.path.join('results', file, "efficiency.svg"))
plt.savefig(os.path.join('results', file, "efficiency.eps"))       # no transparency
# plt.show()
plt.clf()

# Plot of the costs
# Bar chart with zorder=2
bars = plt.bar(range(len(best_strategy_costs)), best_strategy_costs, color='blue', label='Costs', zorder=2)

# Line chart with zorder=3
plt.plot(best_strategy_costs_cumulated, color='red', marker='o', label='Cumulated Costs', zorder=3)

# Grid
plt.grid(True, zorder=1)
plt.grid(which='major', color='#DDDDDD', linewidth=0.9, zorder=1)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9, zorder=1)
plt.minorticks_on()

# Axis title and plot title
plt.xlabel('Time')
plt.ylabel('Costs [EUR]')
plt.title('Costs of the best strategy')

# Show legend
plt.legend()

# Saving the plots in different file formats
plt.savefig(os.path.join('results', file, "costs.png"))
plt.savefig(os.path.join('results', file, "costs.svg"))
plt.savefig(os.path.join('results', file, "costs.eps"))     # no transparency
# plt.show()
plt.clf()

# Plot of the costs logarithmic
# Bar chart with zorder=2
bars = plt.bar(range(len(best_strategy_costs)), best_strategy_costs, color='blue', label='Costs', zorder=2)

# Line chart with zorder=3
plt.plot(best_strategy_costs_cumulated, color='red', marker='o', label='Cumulated Costs', zorder=3)

# Grid
plt.grid(True, zorder=1)
plt.grid(which='major', color='#DDDDDD', linewidth=0.9, zorder=1)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9, zorder=1)
plt.minorticks_on()

# Set y-axis to logarithmic scale
plt.yscale('log')

# Axis title and plot title
plt.xlabel('Time')
plt.ylabel('Costs [EUR]')
plt.title('Costs of the best strategy')

# Show legend
plt.legend()

# Saving the plots in different file formats
plt.savefig(os.path.join('results', file, "costs_log.png"))
plt.savefig(os.path.join('results', file, "costs_log.svg"))
plt.savefig(os.path.join('results', file, "costs_log.eps"))     # no transparency
# plt.show()
plt.clf()
