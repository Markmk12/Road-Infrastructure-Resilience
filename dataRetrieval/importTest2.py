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
G = ox.graph_from_place(region5, network_type='drive', custom_filter=cf1)
G = ox.project_graph(G)
print(G)

# Plot region within a radius
# G = ox.graph_from_address(region1, network_type='drive', dist=1400)
# G = ox.project_graph(G)

ox.plot_graph(G, node_color="r", node_size=4, edge_color="blue", edge_linewidth=1, bgcolor='white', show=False, close=False)
plt.show()

# Save as a shape file for QGis
# G.save_graphml_shapefile(filename=r"C:\Users\markm\Desktop\testshape.shp")

end = time.time()
print(end-start)

# Matrix
roadMatrix = nx.to_numpy_array(G)      # Adjacency matrix
print(roadMatrix)
print(np.size(roadMatrix))             # 2295225

