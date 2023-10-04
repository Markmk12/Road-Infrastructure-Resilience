import numpy as np
import pandas as pd
import copy
import networkx as nx
import matplotlib.pyplot as plt
from traffic import function_lib as tf
from pavement import function_lib as pv
from maintenance import function_lib as ma
from network_efficiency import function_lib as eff

# Notes and TODOs:
# The sampling should not update the graph of time step t
# Wird jede Kante nach jedem Schritt mit dem z.B. PCI Mean überschrieben?????
# ...

# Ideal road network
road_network_0 = nx.Graph()
road_network_0.add_node(1)
road_network_0.add_node(2)
road_network_0.add_node(3)
road_network_0.add_node(4)
road_network_0.add_node(5)
road_network_0.add_edge(1, 2, key=0, length=100, lanes=4, velocity=100, max_speed=100, AAT=450, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(2, 1, key=1, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(2, 3, key=2, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(1, 3, key=3, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(3, 4, key=4, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(2, 4, key=5, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_0.add_edge(4, 5, key=6, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)

# Road network for simulation
road_network_1 = nx.Graph()
road_network_1.add_node(1)
road_network_1.add_node(2)
road_network_1.add_node(3)
road_network_1.add_node(4)
road_network_1.add_node(5)
road_network_1.add_edge(1, 2, key=0, length=100, lanes=4, velocity=100, max_speed=100, AAT=450, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(2, 1, key=1, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(2, 3, key=2, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(1, 3, key=3, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(3, 4, key=4, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(2, 4, key=5, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)
road_network_1.add_edge(4, 5, key=6, length=100, lanes=4, velocity=100, max_speed=100, AAT=700, PCI=100, time=60, maintenance=0, age=0)

# Randomly sampling PCI and age to each edge and adjust correspond velocity and travel time
for _, _, data in road_network_1.edges(data=True):
    data['PCI'] = np.random.choice(list(range(70, 100)))
    data['age'] = np.random.choice(list(range(8)))
    data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'], data['max_speed'])
    data['time'] = tf.travel_time(data['velocity'], data['length'])

# Test
for u, v, attrs in road_network_1.edges(data=True):
    print(f"Edge: ({u}, {v}), Attributes: {attrs}")

# # Visualize the graph
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_size=500)
# labels = nx.get_edge_attributes(G, 'weight')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

# Lists
normed_efficiency_t_samples = []
normed_efficiency_history = []
pci_mean_history = []

# Simulation time period
simulation_time_period = range(1, 100)

# Simulation of the network efficiency over 100 years
for t in simulation_time_period:

    sample_size = 30

    pci_samples = []
    velocity_samples = []
    time_samples = []
    deterioration_samples = []

    pci_samples_new = [[] for _ in range(sample_size)]                      # hier soll für jede Kante ihre gesampelten PCI in einer sublist gespeichert werden

    for _ in range(sample_size):

        # Create a copy of the road network to avoid modifying the original
        temp_network = copy.deepcopy(road_network_1)

        for u, v, data in temp_network.edges(data=True):

            data['PCI'] = data['PCI'] - pv.pavement_deterioration_gamma_process_alternative(data['PCI'], t)
            data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'], data['max_speed'])
            data['time'] = tf.travel_time(data['velocity'], data['length'])
            data['age'] = data['age'] + 1

            # Store samples of the attributes

            pci_samples.append((u, v, data['PCI']))
            # pci_samples.append(data['PCI'])
            velocity_samples.append(data['velocity'])
            time_samples.append(data['time'])

        # Calculations for each sample
        # Sample Network Efficiency
        efficiency_t_sample = eff.network_efficiency(temp_network)

        # Sample Normalizing
        normed_efficiency_t = efficiency_t_sample / eff.network_efficiency(road_network_0)

        # Save the Network Efficiency sample
        normed_efficiency_t_samples.append(normed_efficiency_t)

    # Calculate means of the samples at time t
    PCI_mean = np.mean(pci_samples)                         # Bevor mean berechnen evtl. Liste pci_samples enthält die Tupel der Kanten filtern oder indizieren  siehe Chat GPT!
    velocity_mean = np.mean(velocity_samples)
    time_mean = np.mean(time_samples)

    # Mean of the Network Efficiency at time t
    efficiency_t_mean = np.mean(normed_efficiency_t_samples)

    # Update the road network with the calculated PCI means
    for u, v, data in road_network_1.edges(data=True):
        data['PCI'] = PCI_mean
        data['velocity'] = velocity_mean
        data['time'] = time_mean

    # Saving the means
    pci_mean_history.append(PCI_mean)
    normed_efficiency_history.append(efficiency_t_mean)

print("The predicted normalized Network Efficiency is: " + str(normed_efficiency_history[-1]))
print("The predicted PCI mean is: " + str(pci_mean_history))

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
