import numpy as np
import pandas as pd
import copy
import networkx as nx
import matplotlib.pyplot as plt
from function_library import system, traffic_dynamics as tf, pavement as pv

# Notes and TODOs:
# Code umschreiben für MultiGraph

# Ideal road network
road_network_0 = nx.Graph()
road_network_0.add_node(1)
road_network_0.add_node(2)
road_network_0.add_node(3)
road_network_0.add_node(4)
road_network_0.add_node(5)
road_network_0.add_edge(1, 2, key=0, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=450, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(2, 1, key=1, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(2, 3, key=2, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(1, 3, key=3, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(3, 4, key=4, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(2, 4, key=5, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(4, 5, key=6, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)

# Road network for simulation
road_network_1 = nx.Graph()
road_network_1.add_node(1)
road_network_1.add_node(2)
road_network_1.add_node(3)
road_network_1.add_node(4)
road_network_1.add_node(5)
road_network_1.add_edge(1, 2, key=0, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=450, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(2, 1, key=1, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(2, 3, key=2, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(1, 3, key=3, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(3, 4, key=4, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(2, 4, key=5, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(4, 5, key=6, length=100000, lanes=4, velocity=100, maxspeed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)

# Randomly sampling PCI and age to each edge and adjust correspond velocity and travel time
for _, _, data in road_network_1.edges(data=True):
    data['PCI'] = np.random.choice(list(range(70, 100)))
    data['age'] = np.random.choice(list(range(8)))
    data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'], data['maxspeed'])
    data['time'] = tf.travel_time(data['velocity'], data['length'])

# Test
for u, v, attrs in road_network_1.edges(data=True):
    print(f"Edge: ({u}, {v}), Attributes: {attrs}")

# Visualize the graph
pos = nx.spring_layout(road_network_1)
nx.draw(road_network_1, pos, with_labels=True, node_size=500)
labels = nx.get_edge_attributes(road_network_1, 'PCI')
nx.draw_networkx_edge_labels(road_network_1, pos, edge_labels=labels)
plt.show()

# Lists
normed_efficiency_t_samples = []
normed_efficiency_history = []
pci_mean_history = []

# Simulation time period
simulation_time_period = range(0, 101)                          # 0-101

# Simulation of the network efficiency over 100 years
for t in simulation_time_period:

    sample_size = 5

    pci_samples = []
    velocity_samples = []
    time_samples = []
    deterioration_samples = []

    pci_samples_new = [[] for _ in range(sample_size)]                      # hier soll für jede Kante ihre gesampelten PCI in einer sublist gespeichert werden

    for _ in range(sample_size):

        # Create a copy of the road network to avoid modifying the original
        temp_network = copy.deepcopy(road_network_1)

        for u, v, data in temp_network.edges(data=True):

            data['PCI'] = data['PCI'] - pv.pavement_deterioration_variance_gamma_process(data['PCI'], t)

            if data['PCI'] <= 0:
                data['PCI'] = 10
            elif data['PCI'] > 100:
                data['PCI'] = 100

            data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'], data['maxspeed'])
            data['time'] = tf.travel_time(data['velocity'], data['length'])

            # Store samples of the attributes

            pci_samples.append((u, v, data['PCI']))
            # pci_samples.append(data['PCI'])
            velocity_samples.append(data['velocity'])
            time_samples.append(data['time'])

        # Calculations for each sample
        # Sample Network Efficiency
        efficiency_t_sample = system.network_efficiency(temp_network)

        # Sample Normalizing
        normed_efficiency_t = efficiency_t_sample / system.network_efficiency(road_network_0)

        # Save the Network Efficiency sample
        normed_efficiency_t_samples.append(normed_efficiency_t)

    # Calculate means of the samples at time t          # not needed anymore only for the plot
    PCI_mean = np.mean(pci_samples)
    velocity_mean = np.mean(velocity_samples)
    time_mean = np.mean(time_samples)

    # Mean of the Network Efficiency at time t
    efficiency_t_mean = np.mean(normed_efficiency_t_samples)

    # Mean of the PCI at time t
    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(pci_samples, columns=['Node1', 'Node2', 'PCI'])
    # Group by the first two columns and compute the mean for each group
    mean_values_df = df.groupby(['Node1', 'Node2']).mean().reset_index()
    # Update edges based on mean_values_df
    for _, row in mean_values_df.iterrows():
        road_network_1.add_edge(row['Node1'], row['Node2'], PCI=row['PCI'])

    # Update the rest of the road network with calculated PCI mean of time t
    for u, v, data in road_network_1.edges(data=True):
        data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'], data['maxspeed'])
        data['time'] = tf.travel_time(data['velocity'], data['length'])

        data['age'] = data['age'] + 1

    # Test the road network at time t
    for u, v, attrs in road_network_1.edges(data=True):
        print(f"Edge: ({u}, {v}), Attributes: {attrs}")

    # Saving the means
    pci_mean_history.append(PCI_mean)
    normed_efficiency_history.append(efficiency_t_mean)

# Resilience
resilience = system.resilience_metric(normed_efficiency_history, 1, len(simulation_time_period))

print("The predicted normalized Network Efficiency is: " + str(normed_efficiency_history[-1]))
print("The predicted PCI mean is: " + str(pci_mean_history))
print(print("The resilience is: " + str(resilience)))

# Plotting
plt.step(simulation_time_period, normed_efficiency_history, '-')
plt.xlabel('Time t [Years]')
plt.ylabel('Mean of Network Efficiency [-]')
plt.title('Prediction of Network Efficiency')
plt.grid(True)
plt.grid(which='major', color='#DDDDDD', linewidth=0.9)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9)
plt.minorticks_on()
plt.show()

plt.step(simulation_time_period, pci_mean_history, '-')
plt.xlabel('Time t [Years]')
plt.ylabel('Mean of the PCI [-]')
plt.title('PCI Prediction')
plt.grid(True)
plt.grid(which='major', color='#DDDDDD', linewidth=0.9)
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.9)
plt.minorticks_on()
plt.show()
