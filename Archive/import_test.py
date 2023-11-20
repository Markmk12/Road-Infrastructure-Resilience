import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Coordinates (e.g. center of Berlin lat. 52.5200 / lon. 13.4050)
latitude = 49.6311
longitude = 8.3611
distance = 20 * 1000     # Radius around coordinates in [m] (e.g. Berlin 40 km)

tolerance = 200
# Netzwerk laden und vereinfachen
location = "Berlin"
# cf00 = '["highway"~"motorway|trunk|primary"]'
cf00 = '["highway"~"motorway|trunk|primary|motorway_link|trunk_link|primary_link"]'

G = ox.graph_from_place(location, network_type='drive', custom_filter=cf00)
# G = ox.graph_from_point((latitude, longitude), dist=distance, network_type='drive', custom_filter=cf00)


# Simplification
G = ox.project_graph(G)
G = ox.simplification.consolidate_intersections(G, tolerance=tolerance, rebuild_graph=True, dead_ends=False, reconnect_edges=True)

print(G)

fig, ax = ox.plot_graph(G, node_color='red', edge_color='black', bgcolor='white', show=False, close=False)
plt.show()

# nx.draw(G)
# plt.show()

# MultiGraph in einen einfachen Graphen umwandeln
# G = nx.Graph(G)
# print(G)