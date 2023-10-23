import numpy as np
import networkx as nx
from function_library import system

# G = nx.MultiDiGraph()
# G.add_node(1)
# G.add_node(2)
# G.add_node(3)
# G.add_node(4)
# G.add_node(5)
# G.add_edge(1, 2, key=0, highway='primary', length=100000, capacity=100000, lanes=2, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(2, 1, key=1, highway='primary', length=100000, capacity=100000, lanes=2, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(2, 3, key=2, highway='secondary', length=100000, capacity=15000, lanes=2, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(1, 3, key=3, highway='secondary', length=100000, lanes=1, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(3, 4, key=4, highway='secondary', length=100000, lanes=1, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(2, 4, key=5, highway='primary', length=100000, lanes=2, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(4, 2, key=6, highway='primary', length=100000, lanes=2, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(4, 5, key=7, highway='primary', length=100000, lanes=1, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(5, 4, key=8, highway='primary', length=100000, lanes=1, velocity=100, maxspeed=100, traffic_load=0, PCI=100, time=60, maintenance='no', age=0, duration=0)

imported_road_network = nx.read_gexf("../network_import/networks_of_investigation/ruhr.gexf")

# Perfect state of the road network
road_network_0 = imported_road_network

if nx.is_strongly_connected(road_network_0):
    print("The graph is strongly connected. Continue with the program.")
else:
    print("ERROR: The graph is either weakly connected or not connected.")
    # sys.exit(1)

shortest_path_matrix = nx.floyd_warshall_numpy(road_network_0, weight='time')
print(shortest_path_matrix)

# if np.any(np.isinf(shortest_path_matrix)):
#     print("inf values detected in shortest path matrix!")

optimal_efficiency = system.network_efficiency(road_network_0)
print(optimal_efficiency)

