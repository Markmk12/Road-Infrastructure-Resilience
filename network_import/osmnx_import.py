import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np
import sys
from function_library import traffic_dynamics as tf


# [No. 1] Set single location here (if you are using this comment the next two sections!)
location = 'Hannover'

# [No. 2] Set multiple locations here (if you are using this comment the upper and the following block!)
locations = ['Berlin, Germany', 'Brandenburg, Germany']

# [No. 3] Set coordinates from a location (if you are using this comment the two upper lines)
# Coordinates (e.g. center of Berlin lat. 52.5200 / lon. 13.4050)
latitude = 53.5511
longitude = 9.9937
distance = 8 * 1000     # Radius around coordinates in [m] (e.g. Berlin 40 km)

# Select here which of the above lines [1, 2, 3] should be used for retrieval
retrieve = 1

# Filter (from here no more comment or uncomment something)
filter_0 = ''
filter_1 = '["highway"~"motorway|trunk|primary|motorway_link|trunk_link|primary_link"]'         # Bachelor Thesis
filter_2 = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
filter_3 = '["highway"~"motorway|trunk|primary|secondary"]'
filter_4 = '["highway"~"motorway|trunk|primary"]'
filter_5 = '["highway"~"motorway|primary"]'
filter_6 = '["highway"~"motorway|primary|secondary"]'
filter_7 = '["highway"~"motorway|trunk|primary|secondary|motorway_link|trunk_link|primary_link"]'
filter_8 = '["highway"~"motorway|trunk|primary|secondary"]'

# Set a tolerance in [m] (This is used to combine nearby clusters of nodes of an intersection into one node)
tolerance = 200

#
if retrieve == 1:

    # Get region within its borders
    G = ox.graph_from_place(location, network_type='drive', custom_filter=filter_7)

elif retrieve == 2:

    # Get various cities and regions
    graphs = []
    for city in locations:
        G = ox.graph_from_place(city, network_type='drive', custom_filter=filter_1)
        graphs.append(G)
    # Combine alle single graphs into one
    G = nx.compose_all(graphs)

elif retrieve == 3:
    G = ox.graph_from_point((latitude, longitude), dist=distance, network_type='drive', custom_filter=filter_1)

else:
    print("Please set a single location or a list of multiple locations and comment the line for the other.")
    sys.exit(1)

# Simplification
G = ox.project_graph(G)
G = ox.simplification.consolidate_intersections(G, tolerance=tolerance, rebuild_graph=True, dead_ends=False, reconnect_edges=True)

# Transform MultiDiGraph into MultiGraph
# G = ox.utils_graph.get_undirected(G)

# Filter the graph to remove all roads with the tag "unclassified".
edges_to_remove = [(u, v, k) for u, v, k, data in G.edges(keys=True, data=True) if data.get('highway') == 'unclassified']
G.remove_edges_from(edges_to_remove)

# Removing isolated nodes
G = ox.utils_graph.remove_isolated_nodes(G)

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
print(G)
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


# This code converts the lists to strings by joining the elements of the lists with a comma.
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
# for u, v, key, data in G.edges(data=True, keys=True):
#     print(u, v, key)  # The original key of OSMnx is output here


# Saving the graph
if retrieve == 1:

    # Saving the retrieved graph for export
    nx.write_gexf(G, f"networks_of_investigation/{location.lower()}.gexf")

elif retrieve == 2:

    # Saving the retrieved graph for export
    formatted_cities = [city.replace(", Germany", "").replace(" ", "_").lower() for city in locations]
    filename = "_".join(formatted_cities) + ".gexf"
    nx.write_gexf(G, f"networks_of_investigation/{filename}")

elif retrieve == 3:

    # Saving the retrieved graph for export
    filename = f"graph_{latitude}_{longitude}.gexf"
    nx.write_gexf(G, f"networks_of_investigation/{filename}")

else:
    print("The extracted graph is not stored!")
    sys.exit(1)
