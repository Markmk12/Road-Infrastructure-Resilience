
import osmnx as ox
import matplotlib.pyplot as plt
ox.config(use_cache=True, log_console=True)

place1 = 'Destel, Germany'
place2 = 'Petershagen, Germany'
place3 = 'Hannover, Germany'
place4 = 'Elze, Germany'
place5 = 'Berlin, Germany'
place6 = 'North Rhine-Westphalia, Germany'
place7 = 'Kreis Minden-Lübbecke, Germany'
place8 = 'Rahden,Germany'
place9 = 'Rinteln, Germany'
place10 = 'Springe, Germany'

region1 = 'Region Hannover, Germany'

cf = '["highway"~"motorway|primary|secondary"]'

G = ox.graph_from_place(region1, network_type='drive', custom_filter=cf)
G = ox.project_graph(G)  #G was Gp

#G = ox.simplification.simplify_graph(G, strict=True, remove_rings=True)
G = ox.simplification.consolidate_intersections(G, tolerance=150, rebuild_graph=True, dead_ends=False, reconnect_edges=True)

G = ox.utils_graph.get_undirected(G)
#G.add_edge(1, 15)

fig, ax = ox.plot_graph(G, node_color="r",node_size=20, edge_color="black",edge_linewidth=1,bgcolor='white', show=False, close=False)

plt.savefig('importTestPhoto', dpi =1000,  bbox_inches='tight')
plt.show()