# Main program

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import dataRetrieval as dr

roadNetwork1 = nx.MultiDiGraph()
roadNetwork1.add_node(1)
roadNetwork1.add_node(2)
roadNetwork1.add_edge(1, 2, key=0, length=12, lanes=2, velocity=100, AAT=450, PCI=50, time=15)   # Der key-Parameter hilft dabei, die verschiedenen Kanten zu unterscheiden.
roadNetwork1.add_edge(2, 1, key=1, length=12, lanes=2, velocity=100, AAT=700, PCI=90, time=15)

# nx.draw(roadNetwork1)
# plt.show()

# Alle Kanten mit ihren Attributen als Liste ausgeben
# roadList1 = list(roadNetwork1.edges(data=True, keys=True))
# print(roadList1)


def velocity_change(PCI, velocity):
    delta_v = 1385.406/PCI-15.985
    velocity = 100                      # das deltaV bezieht sich immer auf maxSpeed
    new_velocity = velocity - delta_v
    return new_velocity


def PCI_degradation(PCI):
    degradedPCI = PCI-0.1
    return degradedPCI

velocityHistory = []


# Simulation
# Alle Kanten durchlaufen und das Attribut traffic_strength verändern und das 360 mal
for j in range(360):
    for u, v, key, data in roadNetwork1.edges(data=True, keys=True):
        data['PCI'] = PCI_degradation(data['PCI'])
        data['velocity'] = velocity_change(data['PCI'], data['velocity'])

    # Werte der Kante (1, 2, 0) speichern
    velocityHistory.append(roadNetwork1[1][2][0]['velocity'])

# Plotten einer Kantengeschwindigkeit
plt.plot(range(360), velocityHistory, color='tab:red')
plt.xlabel('Iteration')
plt.ylabel('velocity')
plt.title("Verlauf der Geschwindigkeit v [km/h]")
plt.show()

# Alle Kanten mit ihren Attributen als Liste ausgeben
roadList1 = list(roadNetwork1.edges(data=True, keys=True))
print(roadList1)
