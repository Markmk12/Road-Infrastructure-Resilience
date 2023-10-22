import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

# Netzwerk laden und vereinfachen
location = "Hameln"
cf00 = '["highway"~"motorway|trunk|primary|motorway_link|trunk_link|primary_link"]'
G = ox.graph_from_place(location, network_type='drive', custom_filter=cf00)
# G_simplified = ox.simplify_graph(G)

# MultiGraph in einen einfachen Graphen umwandeln
G = nx.Graph(G)
print(G)

edge_list = G.edges(data=True)
# Print the edge list and attributes
for u, v, attr in edge_list:
    print(f"Edge ({u}, {v}): {attr}")

# fig, ax = ox.plot_graph(G, node_color='red', edge_color='black', bgcolor='white', show=False, close=False)
# plt.show()

nx.draw(G)
plt.show()
