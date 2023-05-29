import time
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox
import numpy as np

region1 = 'Hannover, Germany'
region2 = 'Springe, Germany'
region3 = 'Rinteln, Germany'
region4 = 'Emskirchen, Germany'
region5 = 'Belgrade, Serbia'        # 21 sec. on laptop
region6 = 'Amed, Turkey'            # 45 sec. on laptop

state1 = 'Luxembourg'
state2 = 'Germany'
# state3 = 'Russia'
state4 = 'Turkey'
state5 = 'Austria'  # ca. 65 sec. on laptop
cf1 = '["highway"~"motorway|primary|secondary"]'
cf2 = '["highway"~"motorway"]'

start = time.time()

# Plot region within its borders
G1 = ox.graph_from_place(region5, network_type='drive', custom_filter=cf1)
G1 = ox.project_graph(G1)
print(G1)


# Simplification
G2 = ox.simplification.consolidate_intersections(G1, tolerance=150, rebuild_graph=True, dead_ends=False, reconnect_edges=True)
G2 = ox.utils_graph.get_undirected(G2)

# Plot
ox.plot_graph(G1, node_color="r", node_size=4, edge_color="blue", edge_linewidth=1, bgcolor='white', show=False, close=False)
plt.show()
ox.plot_graph(G2, node_color="r", node_size=4, edge_color="blue", edge_linewidth=1, bgcolor='white', show=False, close=False)
plt.show()

# Matrix
roadMatrix = nx.to_numpy_array(G1)      # Adjacency matrix
print(roadMatrix)
print(np.size(roadMatrix))             # 2295225

end = time.time()
print(end-start)