import math
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np

# Filter
cf01 = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
cf00 = '["highway"~"motorway|trunk|primary"]'                       # diesen filter für autobahnen-betrachtung
cf1 = '["highway"~"motorway|trunk|primary|secondary"]'
cf2 = '["highway"~"motorway|trunk|primary"]'
cf3 = '["highway"~"motorway"]'

# Plot region within its borders
G = ox.graph_from_place('Niedersachsen', network_type='drive', custom_filter=cf00)
# print(G)

# Simplification
G = ox.project_graph(G)
G = ox.simplification.consolidate_intersections(G, tolerance=10, rebuild_graph=True, dead_ends=False, reconnect_edges=True)

# Transform MultiDiGraph into MultiGraph
G = ox.utils_graph.get_undirected(G)
print(G)

# Delete nodes with degree of 2 or lower
# nodes_to_remove = [node for node, degree in G.degree() if degree == 2]
# G.remove_nodes_from(nodes_to_remove)


# Matrix
roadMatrix = nx.to_numpy_array(G)      # Adjacency matrix
roadMatrixElements = np.size(roadMatrix)
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
