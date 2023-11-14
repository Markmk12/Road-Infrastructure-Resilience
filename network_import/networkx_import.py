import networkx as nx
import matplotlib as plt

# Here you can create simple graphs for testing
# The graphs must be MultiDiGraphs or MultiGraphs (Graphs or DiGraphs will not work in the main program)
G = nx.MultiDiGraph()
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)
G.add_node(5)
G.add_edge(1, 2, key=0, length=100000, lanes=2, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(1, 2, key=1, length=100000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(2, 1, key=2, length=100000, lanes=2, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(2, 3, key=3, length=100000, lanes=2, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(1, 3, key=4, length=100000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(3, 4, key=5, length=100000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(2, 4, key=6, length=100000, lanes=2, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(4, 2, key=7, length=100000, lanes=2, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(4, 5, key=8, length=100000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(5, 4, key=9, length=100000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)

nx.draw(G)


# Saving
# nx.write_gexf(G, "networks_of_investigation/simple_test_graphs/simple_test_graph_1.gexf")