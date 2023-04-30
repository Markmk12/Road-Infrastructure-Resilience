import time
import matplotlib.pyplot as plt
import osmnx as ox

region1 = 'Hannover, Germany'
region2 = 'Springe, Germany'
region3 = 'Rinteln, Germany'
region4 = 'Emskirchen, Germany'

state1 = 'Luxembourg'
state2 = 'Germany'
# state3 = 'Russia'
state4 = 'Turkey'

cf1 = '["highway"~"motorway|primary|secondary"]'
cf2 = '["highway"~"motorway"]'

start = time.time()

# Plot region within its borders
G = ox.graph_from_place(state4, network_type='drive', custom_filter=cf2)
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

# Art plot
# ox.plot_graph(G, node_color="white", node_size=4, edge_color="white", edge_linewidth=1, bgcolor='black', show=False, close=False)
# plt.show()