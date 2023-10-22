import math
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
from function_library import traffic_dynamics as tf

# Set location here:
location = 'Hamburg'

# Filter
cf01 = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
cf00 = '["highway"~"motorway|trunk|primary|motorway_link|trunk_link|primary_link"]'                      # diesen filter für autobahnen-betrachtung
cf1 = '["highway"~"motorway|trunk|primary|secondary"]'
cf2 = '["highway"~"motorway|trunk|primary"]'
cf3 = '["highway"~"motorway"]'

# Plot region within its borders
G = ox.graph_from_place(location, network_type='drive', custom_filter=cf00)               # , custom_filter=cf01)

# G = ox.graph.graph_from_address(52.519514655923146, 13.406701005419093, dist=40000, dist_type='bbox', network_type='drive', custom_filter=cf00)
# G = ox.graph.graph_from_point(52.519514655923146, 13.406701005419093, dist=40000, dist_type='bbox', network_type='drive', custom_filter=cf00)      # Für Berlin betrachtung der Ringautobahn (40km) A10 in Brandenburg notwendig
# print(G)

# Simplification
G = ox.project_graph(G)
G = ox.simplification.consolidate_intersections(G, tolerance=500, rebuild_graph=True, dead_ends=False, reconnect_edges=True)    # Toleranz von 10m oder Toleranz von 150m (Reduktion um fast 50% der Knoten und Kanten) ?

# Transform MultiDiGraph into MultiGraph
# G = ox.utils_graph.get_undirected(G)
print(G)

# Delete nodes with degree of 2 or lower
# nodes_to_remove = [node for node, degree in G.degree() if degree == 2]
# G.remove_nodes_from(nodes_to_remove)

# Filter the graph to remove all roads with the tag "unclassified".
edges_to_remove = [(u, v, k) for u, v, k, data in G.edges(keys=True, data=True) if data.get('highway') == 'unclassified']
G.remove_edges_from(edges_to_remove)

# Removing isolated nodes
graph = ox.utils_graph.remove_isolated_nodes(G)

# Traverses all edges and changes attributes from string to integer
# If an edge has more than one lane data, only the lower value is taken
# If an edge has more than one maxspeed data, the mean is taken

attributes_to_convert = ['lanes', 'maxspeed', 'length']

for u, v, k, data in G.edges(data=True, keys=True):
    for attribute in attributes_to_convert:
        if attribute in data:
            # If the attribute is a single string, convert it to int
            if isinstance(data[attribute], str):
                try:
                    data[attribute] = int(data[attribute])
                except ValueError:
                    pass
            # Debugging
            # elif isinstance(data[attribute], list):
            #     try:
            #         values_as_int = np.array(data[attribute], dtype=int)
            #     except ValueError:
            #         print(f"Failed to convert {data[attribute]} for edge ({u}, {v}, {k})")
            # If the attribute is a list of strings, process as before
            elif isinstance(data[attribute], list):
                try:
                    # Remove non-numeric values from the list
                    valid_values = [int(val) for val in data[attribute] if val.isdigit()]

                    # If there are no valid values, set a default value
                    if not valid_values:
                        data[attribute] = 75  # or another default value
                    else:
                        # If there are valid values, calculate the average
                        if attribute == 'lanes':
                            # Take the minimum value for 'lanes'.
                            data[attribute] = min(valid_values)
                        elif attribute == 'maxspeed':
                            # Calculate the average value for 'maxspeed
                            data[attribute] = int(np.mean(valid_values))
                except ValueError:
                    pass


# Delete attributes
attributes_to_remove = ['u_original', 'v_original', 'from', 'to', 'oneway', 'reversed', 'geometry', 'bridge', 'osmid', 'ref', 'name', 'width', 'highway']
for u, v, k, data in G.edges(data=True, keys=True):
    for attribute in attributes_to_remove:
        data.pop(attribute, None)


# Adding new attributes (for an ideal network)
key_counter = 0
for u, v, k, data in G.edges(data=True, keys=True):
    data['PCI'] = 100
    data['maintenance'] = 'no'
    data['key'] = key_counter
    key_counter += 1

    # Does a 'maxspeed' data exist for the edge?
    if 'maxspeed' in data:
        # If the value is 'none', set it to the default value
        if data['maxspeed'] == 'none':
            data['maxspeed'] = 75
        else:
            try:
                data['maxspeed'] = int(data['maxspeed'])
            except ValueError:
                data['maxspeed'] = 75
    else:
        # If no 'maxspeed' attribute is present, set the default value
        data['maxspeed'] = 75

    data['velocity'] = data['maxspeed']
    data['age'] = 0
    data['time'] = tf.travel_time(data['velocity'], data['length'])


# Get the list of edges and their attributes
edge_list = G.edges(data=True)
# Print the edge list and attributes
for u, v, attr in edge_list:
    print(f"Edge ({u}, {v}): {attr}")

# Visualize the graph as with OSMNX
fig, ax = ox.plot_graph(G, node_color='red', edge_color='black', bgcolor='white', show=False, close=False)
plt.show()

# Save the plot
# fig.savefig(f"plots/networks/{location}.png")

# Visualize the graph as with OSMNX and its attribute PCI
# edge_labels = {(u, v): data['PCI'] for u, v, key, data in G.edges(keys=True, data=True)}
# fig, ax = ox.plot_graph(G, show=False, close=False)
# plt.show()

# Edge labels
# for (u, v), label in edge_labels.items():
#     x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
#     x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
#     ax.text((x1 + x2) / 2, (y1 + y2) / 2, str(label), fontsize=8, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))
# plt.show()

# Visualize the graph as with NetworkX  (Not suitable for drawing MultiGraphs; use OSMNX )
# pos = nx.spring_layout(G)
# nx.draw(G, pos, with_labels=True, node_size=500)
# labels = nx.get_edge_attributes(G, 'PCI')
# nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
# plt.show()

# This code converts the lists to strings by joining the elements of the lists with a comma.
# After running this code, you should be able to save the graph in GEXF format without errors.
# save it.

for u, v, data in G.edges(data=True):
    if isinstance(data.get('osmid'), list):
        data['osmid'] = ",".join(map(str, data['osmid']))
    if isinstance(data.get('name'), list):
        data['name'] = ",".join(data['name'])
    if isinstance(data.get('ref'), list):
        data['ref'] = ",".join(data['ref'])

# check all edge attributes and convert lists to strings
for u, v, data in G.edges(data=True):
    for key, value in data.items():
        if isinstance(value, list):
            data[key] = ",".join(map(str, value))

# Debugging origin keys
for u, v, key, data in G.edges(data=True, keys=True):
    print(u, v, key)  # The original key of OSMnx is output here

# Saving the retrieved graph for export
nx.write_gexf(G, f"networks_of_investigation/{location.lower()}.gexf")
