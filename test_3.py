import numpy as np
import copy
import networkx as nx
import matplotlib.pyplot as plt
from traffic import function_lib as tf
from pavement import function_lib as pv
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
road_network_0.add_edge(1, 2, key=0, length=100, lanes=4, velocity=100, AAT=450, PCI=100, time=60)   # Der key-Parameter hilft dabei, die verschiedenen Kanten zu unterscheiden.
road_network_0.add_edge(2, 1, key=1, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)
road_network_0.add_edge(2, 3, key=2, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)
road_network_0.add_edge(1, 3, key=3, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)
road_network_0.add_edge(3, 4, key=4, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)
road_network_0.add_edge(2, 4, key=5, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)
road_network_0.add_edge(4, 5, key=6, length=100, lanes=4, velocity=100, AAT=700, PCI=100, time=60)

# Road network for simulation
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


# Lists
normed_efficiency_t_samples = []
normed_efficiency_history = []
pci_mean_history = []

# Simulation time period
simulation_time_period = range(1, 70)

# Simulation of the network efficiency over 100 years
for t in simulation_time_period:
    PCI_samples = []
    velocity_samples = []
    time_samples = []

    for _ in range(300):

        # Create a copy of the road network to avoid modifying the original
        temp_network = copy.deepcopy(road_network_1)

        for u, v, data in temp_network.edges(data=True):
            # Simulation for every edge
            data['PCI'] = 100 - pv.pavement_deterioration_gamma_process_alternative(data['PCI'], t)
            data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'])
            data['time'] = tf.travel_time(data['velocity'], data['length'])

            # Store samples of the attributes
            PCI_samples.append(data['PCI'])
            velocity_samples.append(data['velocity'])
            time_samples.append(data['time'])

        # Saving the results for each sample
        # Network Efficiency
        efficiency_t_sample = eff.network_efficiency(temp_network)
        # Normalizing
        normed_efficiency_t = efficiency_t_sample / eff.network_efficiency(road_network_0)
        #
        normed_efficiency_t_samples.append(normed_efficiency_t)

    # Calculate means of the samples at time t
    PCI_mean = np.mean(PCI_samples)
    velocity_mean = np.mean(velocity_samples)
    time_mean = np.mean(time_samples)
    efficiency_t_mean = np.mean(normed_efficiency_t_samples)

    # Update the road network with the calculated means
    for _, _, data in road_network_1.edges(data=True):
        data['PCI'] = PCI_mean
        data['velocity'] = velocity_mean
        data['time'] = time_mean

    # Saving the means
    pci_mean_history.append(PCI_mean)
    normed_efficiency_history.append(efficiency_t_mean)

print("The predicted normalized Network Efficiency is: " + str(normed_efficiency_history[-1]))

# Plotting
plt.plot(simulation_time_period, normed_efficiency_history, '-')
plt.xlabel('Time t [Years]')
plt.ylabel('Mean of Network Efficiency [-]')
plt.title('Prediction of Network Efficiency')
plt.grid(True)
plt.show()
