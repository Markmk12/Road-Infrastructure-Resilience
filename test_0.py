import matplotlib.pyplot as plt
import numpy as np
# my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# sections = np.array_split(my_list, 3)
# print(sections)
#
# [(1, 2, -275.87914433907014), (2, 3, -300.150634997869), (2, 3, -320.8835872200407), (1, 2, -282.87104894897345)]


# Parameter environment (natural deterioration???)
# alpha_environment = 1  # shape minimum 2 so that it starts by 0
# beta_environment = 0.5  # rate

# Parameter traffic load
# alpha_traffic = 1
# beta_traffic = 0.7
#
# time_range = range(0, 51)
#
# for t in time_range:
#     deterioration_delta = np.random.gamma(alpha_environment * t, beta_environment) + np.random.gamma(alpha_traffic * t,
#                                                                                                  beta_traffic)
#     print(deterioration_delta)


# All combinations of strategy paths
import itertools


# quality_levels = ["none", "moderate", "extensive"]

# Erzeuge alle möglichen Tupel für einen Zeitpunkt
# tuples_for_one_timepoint = list(itertools.product(quality_levels, repeat=2))

# Erzeuge alle möglichen Pfade für 4 Zeitpunkte
# all_paths = list(itertools.product(tuples_for_one_timepoint, repeat=4))
#
# print(all_paths)

# Changing the strategy configuration (tuple) every 10 years
# if t == 0:
#     quality_level = strategy[0]
# if t == 10:
#     quality_level = strategy[1]
# if t == 20:
#     quality_level = strategy[2]
# if t == 30:
#     quality_level = strategy[3]


import networkx as nx
import matplotlib.pyplot as plt
import random

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
road_network_0.add_edge(3, 5, key=9, highway='primary', length=100000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)
road_network_0.add_edge(5, 3, key=10, highway='primary', length=100000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)

# Füge 50 zusätzliche Knoten hinzu
for i in range(6, 56):  # Wir starten von 6, da 5 der letzte Knoten in Ihrer Vorlage war
    road_network_0.add_node(i)

# Füge zufällige Kanten hinzu
for _ in range(200):  # Als Beispiel fügen wir 200 zufällige Kanten hinzu
    source = random.randint(1, 55)  # Wählen Sie zufällig einen Knoten von 1 bis 55
    target = random.randint(1, 55)  # Wählen Sie zufällig einen Knoten von 1 bis 55

    # Überprüfen, ob eine Kante zwischen source und target existiert
    if target in road_network_0[source]:
        key = len(road_network_0[source][target])
    else:
        key = 0  # Wenn es keine Kante gibt, setzen wir den key auf 0

    road_network_0.add_edge(source, target, key=key, highway='primary', length=100000, capacity=100000, lanes=4, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0)


print(road_network_0)
pos = nx.spring_layout(road_network_0)
nx.draw(road_network_0, pos, with_labels=True, node_size=500, node_color='skyblue', font_weight='bold')
plt.show()
