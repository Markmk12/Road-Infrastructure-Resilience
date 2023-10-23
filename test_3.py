import osmnx as ox

# Längen- und Breitengrad von Berlin-Mitte
latitude = 52.5200
longitude = 13.4050

# Radius in Metern (50 km)
distance = 40 * 1000

filter_1 = '["highway"~"motorway|trunk|primary|motorway_link|trunk_link|primary_link"]'
tolerance = 400

# Graphen für den angegebenen Umkreis von Berlin-Mitte extrahieren
G = ox.graph_from_point((latitude, longitude), dist=distance, network_type='drive', custom_filter=filter_1)

G = ox.project_graph(G)
G = ox.simplification.consolidate_intersections(G, tolerance=tolerance, rebuild_graph=True, dead_ends=False, reconnect_edges=True)

# Optional: Den Graphen anzeigen
print(G)
ox.plot_graph(G)