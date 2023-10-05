import math
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
from function_library import traffic_dynamics as tf

# Filter
cf01 = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
cf00 = '["highway"~"motorway|trunk|primary"]'                                                                           # diesen filter für autobahnen-betrachtung
cf1 = '["highway"~"motorway|trunk|primary|secondary"]'
cf2 = '["highway"~"motorway|trunk|primary"]'
cf3 = '["highway"~"motorway"]'

# Plot region within its borders
G = ox.graph_from_place('Hameln', network_type='drive', custom_filter=cf1)

# G = ox.graph.graph_from_address(52.519514655923146, 13.406701005419093, dist=40000, dist_type='bbox', network_type='drive', custom_filter=cf00)
# G = ox.graph.graph_from_point(52.519514655923146, 13.406701005419093, dist=40000, dist_type='bbox', network_type='drive', custom_filter=cf00)      # Für Berlin betrachtung der Ringautobahn (40km) A10 in Brandenburg notwendig
# print(G)

# Simplification
G = ox.project_graph(G)
G = ox.simplification.consolidate_intersections(G, tolerance=150, rebuild_graph=True, dead_ends=False, reconnect_edges=True)    # Toleranz von 10m oder Toleranz von 150m (Reduktion um fast 50% der Knoten und Kanten) ?

# Transform MultiDiGraph into MultiGraph
G = ox.utils_graph.get_undirected(G)                                                                                  # Why transform MultiDiGraph into MultiGraph ?
print(G)

# Delete nodes with degree of 2 or lower
# nodes_to_remove = [node for node, degree in G.degree() if degree == 2]
# G.remove_nodes_from(nodes_to_remove)


# Matrix
# roadMatrix = nx.to_numpy_array(G)      # Adjacency matrix
# roadMatrixElements = np.size(roadMatrix)
# number_of_nodes = math.sqrt(roadMatrixElements)
# print(roadMatrixElements)                   # 527712784
# print(math.sqrt(roadMatrixElements))        # 22972.0


# Traverses all edges and changes attributes from string to integer
attributes_to_convert = ['lanes', 'maxspeed']

for u, v, k, data in G.edges(data=True, keys=True):
    for attribute in attributes_to_convert:
        if attribute in data and isinstance(data[attribute], str):
            try:
                data[attribute] = int(data[attribute])
            except ValueError:
                pass   # If the conversion is not possible (e.g. if the string contains not only digits), it will be skipped


# Delete attributes
attributes_to_remove = ['u_original', 'v_original', 'from', 'to', 'oneway', 'reversed', 'geometry', 'osmid']

for u, v, k, data in G.edges(data=True, keys=True):
    for attribute in attributes_to_remove:
        data.pop(attribute, None)


# Adding new attributes
for u, v, k, data in G.edges(data=True, keys=True):
    data['PCI'] = np.random.choice(list(range(70, 100)))
    data['maintenance'] = 0
    data['AAT'] = 700
    data['age'] = np.random.choice(list(range(8)))
    # data['velocity'] = tf.velocity_change(data['PCI'], data['velocity'], data['maxspeed'])
    # data['time'] = tf.travel_time(data['velocity'], data['length'])


# Get the list of edges and their attributes
edge_list = G.edges(data=True)
# Print the edge list and attributes
for u, v, attr in edge_list:
    print(f"Edge ({u}, {v}): {attr}")

# Visualize the graph as with OSMNX
# fig, ax = ox.plot_graph(G, node_color="r", node_size=20, edge_color="black", edge_linewidth=1, bgcolor='white', show=False, close=False)

edge_labels = {(u, v): data['PCI'] for u, v, key, data in G.edges(keys=True, data=True)}
fig, ax = ox.plot_graph(G, show=False, close=False)

# Edge labels
for (u, v), label in edge_labels.items():
    x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
    x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
    ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(label), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
plt.show()
# plt.savefig('photo 1', dpi=1000,  bbox_inches='tight')


# Visualize the graph as with NetworkX  (Not suitable for drawing MultiGraphs; use OSMNX )
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_size=500)
# labels = nx.get_edge_attributes(G, 'PCI')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()
