import math
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd

# Filter
cf01 = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
cf00 = '["highway"~"motorway|trunk|primary"]'                                                                           # diesen filter für autobahnen-betrachtung
cf1 = '["highway"~"motorway|trunk|primary|secondary"]'
cf2 = '["highway"~"motorway|trunk|primary"]'
cf3 = '["highway"~"motorway"]'

# Plot region within its borders
G = ox.graph_from_place('Hannover', network_type='drive', custom_filter=cf00)
# G = ox.graph.graph_from_address(52.519514655923146, 13.406701005419093, dist=40000, dist_type='bbox', network_type='drive', custom_filter=cf00)
# G = ox.graph.graph_from_point(52.519514655923146, 13.406701005419093, dist=40000, dist_type='bbox', network_type='drive', custom_filter=cf00)      # Für Berlin betrachtung der Ringautobahn (40km) A10 in Brandenburg notwendig
# print(G)

# Simplification
G = ox.project_graph(G)
G = ox.simplification.consolidate_intersections(G, tolerance=150, rebuild_graph=True, dead_ends=False, reconnect_edges=True)    # Toleranz von 10m oder Toleranz von 150m (Reduktion um fast 50% der Knoten und Kanten) ?

# Transform MultiDiGraph into MultiGraph
# G = ox.utils_graph.get_undirected(G)                                                                                  # Why transform MultiDiGraph into MultiGraph ?
print(G)

# Delete nodes with degree of 2 or lower
# nodes_to_remove = [node for node, degree in G.degree() if degree == 2]
# G.remove_nodes_from(nodes_to_remove)


# Matrix
roadMatrix = nx.to_numpy_array(G)      # Adjacency matrix
roadMatrixElements = np.size(roadMatrix)
number_of_nodes = math.sqrt(roadMatrixElements)
print(roadMatrixElements)                   # 527712784
print(math.sqrt(roadMatrixElements))        # 22972.0

# Get the list of edges and their attributes
edge_list = G.edges(data=True)
# Print the edge list and attributes
for u, v, attr in edge_list:
    print(f"Edge ({u}, {v}): {attr}")

# Plot
fig, ax = ox.plot_graph(G, node_color="r", node_size=20, edge_color="black",edge_linewidth=1,bgcolor='white', show=False, close=False)
plt.show()
# plt.savefig('photo 1', dpi =1000,  bbox_inches='tight')