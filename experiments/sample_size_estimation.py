import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
from function_library import system, traffic_dynamics as tf, pavement as pv, maintenance as ma
import itertools
import time
import os
import sys


# Import a road network (You can find examples in: network_import/networks_of_investigation)
# imported_road_network = nx.read_gexf("network_import/networks_of_investigation/graph_51.2277_6.7735.gexf")
imported_road_network = nx.read_gexf("simple_test_graph_1.gexf")

# Perfect state of the road network
road_network_0 = imported_road_network
print(road_network_0)

# Ideal network efficiency (optimal efficiency)
optimal_efficiency = system.network_efficiency(road_network_0)
print(optimal_efficiency)

# Import of a graph
road_network_1 = road_network_0

# Randomly sampling PCI and age to each edge and adjust correspond velocity and travel time
# The value of the PCI depends on the age

for _, _, key, data in road_network_1.edges(keys=True, data=True):

    data['age'] = np.random.choice(list(range(20)))                             # 14  oder # 18

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


quality_levels = ["sparse", "moderate", "extensive"]
# strategy = [('extensive', 'moderate'), ('extensive', 'sparse'), ('moderate', 'sparse')]
# strategy = [('moderate', 'moderate'), ('moderate', 'moderate'), ('moderate', 'moderate')]
strategy = [('sparse', 'sparse'), ('sparse', 'sparse'), ('sparse', 'sparse')]


simulation_time_period = range(0, 101)
sample_sizes = [5, 25, 50, 100, 200, 400, 800, 1600]

samples_matrix_efficiency = np.zeros((len(sample_sizes), len(simulation_time_period)))
samples_matrix_variance = np.zeros((len(sample_sizes), len(simulation_time_period)))
samples_matrix_standard = np.zeros((len(sample_sizes), len(simulation_time_period)))

for idx, sample_size in enumerate(sample_sizes):

    # Matrix
    efficiency_matrix = np.zeros((sample_size, len(simulation_time_period)))

    # Simulation of the network efficiency over 100 years
    for sample in range(sample_size):

        # Create a copy of the road network to avoid modifying the original
        temp_network = copy.deepcopy(road_network_1)

        # start2 = time.time()

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

                    # Count this preventive measure
                    # prev_measures_count_matrix[sample, t] = prev_measures_count_matrix[sample, t] + 1

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

                    # Count this preventive measure
                    # corr_measures_count_matrix[sample, t] = corr_measures_count_matrix[sample, t] + 1

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
                    # costs_matrix[sample, t] = costs_matrix[sample, t] + costs

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
                    # costs_matrix[sample, t] = costs_matrix[sample, t] + costs

            # Network Efficiency at time t
            efficiency_sample_t = system.network_efficiency(temp_network)
            # Sample Normalizing
            normed_sample_efficiency_t = efficiency_sample_t / optimal_efficiency
            # Save the normed efficiency at time t in a matrix (rows = sample, columns = time)
            efficiency_matrix[sample, t] = normed_sample_efficiency_t

            # end3 = time.time()
            # print("Execution time of one time step: ", str(end3 - start3), "[sec]")

        # end2 = time.time()
        # print("Execution time of one sample: ", str(end2 - start2), "[sec]")

    # Delete all rows (sample) in the matrix that have a row element greater than 1
    # efficiency_matrix = efficiency_matrix[~(efficiency_matrix > 1).any(axis=1)]

    # Calculate the efficiency mean of each column and save it in an extra row
    mean_efficiency_row = efficiency_matrix.mean(axis=0)
    variance_efficiency_row = efficiency_matrix.var(axis=0)
    standard_efficiency_row = efficiency_matrix.std(axis=0)

    efficiency_matrix = np.vstack([efficiency_matrix, mean_efficiency_row])

    # Calculate the costs mean of each column and save it in an extra row
    # mean_costs_row = costs_matrix.mean(axis=0)
    # costs_matrix = np.vstack([costs_matrix, mean_costs_row])
    # costs_history_matrix[idx] = mean_costs_row

    # Calculate the mean of the preventive measures count of each column
    # mean_prev_row = prev_measures_count_matrix.mean(axis=0)
    # prev_measures_count_matrix = np.vstack([prev_measures_count_matrix, mean_prev_row])
    # preventive_history_matrix[idx] = mean_prev_row

    # Calculate the mean of the corrective measures count of each column
    # mean_corr_row = corr_measures_count_matrix.mean(axis=0)
    # corr_measures_count_matrix = np.vstack([corr_measures_count_matrix, mean_corr_row])
    # corrective_history_matrix[idx] = mean_corr_row

    # Resilience
    # resilience = system.resilience_metric(efficiency_matrix[-1, :], len(simulation_time_period))
    # strategies_matrix_resilience[idx] = resilience

    # Save the estimated efficiency es an entry of strategies_matrix_efficiency
    samples_matrix_efficiency[idx, :] = mean_efficiency_row
    samples_matrix_variance[idx, :] = variance_efficiency_row
    samples_matrix_standard[idx, :] = standard_efficiency_row


# Plot Variance
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(samples_matrix_variance.shape[0]):
    ax.plot(samples_matrix_variance[i], label=f'{sample_sizes[i]} Samples')

# ax.set_title('Variance of Each Sample Over Time')
ax.set_xlabel('Time')
ax.set_ylabel('Variance')
ax.legend(loc='best', fontsize='small')

# Grid
plt.grid(True)
plt.grid(which='major', color='#DDDDDD', linewidth=0.9, zorder=1)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9, zorder=1)
plt.minorticks_on()

# Saving Plot
plt.tight_layout()
plt.savefig('variance_plot.png')
plt.savefig('variance_plot.eps', format='eps')
plt.savefig('variance_plot.svg', format='svg')

plt.show()


# Standard Deviation in %
samples_matrix_standard_percent = samples_matrix_standard * 100

fig, ax = plt.subplots(figsize=(10, 6))
for i in range(samples_matrix_standard_percent.shape[0]):
    ax.plot(samples_matrix_standard_percent[i], label=f'{sample_sizes[i]} Samples')

# ax.set_title('Standard Deviation of Each Sample Over Time (in %)')
ax.set_xlabel('Time')
ax.set_ylabel('Standard Deviation (%)')
ax.legend(loc='best', fontsize='small')

# Grid
plt.grid(True)
plt.grid(which='major', color='#DDDDDD', linewidth=0.9, zorder=1)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9, zorder=1)
plt.minorticks_on()

# Saving
plt.tight_layout()
plt.savefig('standard_deviation_percent_plot.png')
plt.savefig('standard_deviation_percent_plot.eps', format='eps')
plt.savefig('standard_deviation_percent_plot.svg', format='svg')

plt.show()


# Plot Efficiency
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(samples_matrix_efficiency.shape[0]):
    ax.plot(samples_matrix_efficiency[i], label=f'{sample_sizes[i]} Samples')

# ax.set_title('Efficiency of Each Sample Over Time')
ax.set_xlabel('Time')
ax.set_ylabel('Efficiency')
ax.legend(loc='best', fontsize='small')

# Grid
plt.grid(True)
plt.grid(which='major', color='#DDDDDD', linewidth=0.9, zorder=1)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9, zorder=1)
plt.minorticks_on()

# Saving Plot
plt.tight_layout()
plt.savefig('efficiency_plot.png')
# plt.savefig('efficiency_plot.eps', format='eps')
plt.savefig('efficiency_plot.svg', format='svg')

plt.show()
