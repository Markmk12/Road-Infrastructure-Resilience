# Main program

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from traffic import function_lib as tf
from pavement import function_lib as pv

road_network_1 = nx.MultiDiGraph()
road_network_1.add_node(1)
road_network_1.add_node(2)
road_network_1.add_edge(1, 2, key=0, length=12, lanes=2, velocity=100, AAT=450, PCI=100, time=15)   # Der key-Parameter hilft dabei, die verschiedenen Kanten zu unterscheiden.
road_network_1.add_edge(2, 1, key=1, length=12, lanes=2, velocity=100, AAT=700, PCI=100, time=15)

# nx.draw(road_network_1)
# plt.show()

# Alle Kanten mit ihren Attributen als Liste ausgeben
# road_list_1 = list(road_network_1.edges(data=True, keys=True))
# print(road_list_1)

velocity_history = []

# Simulation
# Alle Kanten durchlaufen und das Attribut traffic_strength verändern und das 360 mal
for j in range(100):
    for u, v, key, data in road_network_1.edges(data=True, keys=True):
        data['PCI'] = pv.pavement_deterioration(data['PCI'])
        data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'])

    # Werte der Kante (1, 2, 0) speichern
    velocity_history.append(road_network_1[1][2][0]['velocity'])

# Plotten einer Kantengeschwindigkeit
plt.plot(range(100), velocity_history, color='tab:red')
plt.xlabel('Iteration')
plt.ylabel('velocity')
plt.title("Verlauf der Geschwindigkeit v [km/h]")
plt.show()

# Alle Kanten mit ihren Attributen als Liste ausgeben
road_list_1 = list(road_network_1.edges(data=True, keys=True))
print(road_list_1)
