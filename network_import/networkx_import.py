import networkx as nx
import matplotlib.pyplot as plt

# Here you can create simple graphs for testing
# The graphs must be MultiDiGraphs or MultiGraphs (Graphs or DiGraphs will not work in the main program)
# G = nx.MultiDiGraph()
# G.add_node(1)
# G.add_node(2)
# G.add_node(3)
# G.add_node(4)
# G.add_node(5)
# G.add_edge(1, 2, key=0, length=100000, lanes=2, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(1, 2, key=1, length=100000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(2, 1, key=2, length=100000, lanes=2, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(2, 3, key=3, length=100000, lanes=2, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(1, 3, key=4, length=100000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(3, 4, key=5, length=100000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(2, 4, key=6, length=100000, lanes=2, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(4, 2, key=7, length=100000, lanes=2, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(4, 5, key=8, length=100000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
# G.add_edge(5, 4, key=9, length=100000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)

G = nx.MultiDiGraph()

# Kanten hinzufügen basierend auf dem TikZ-Netzwerk
# Knoten hinzufügen
G.add_node(1)  # entspricht a im TikZ-Netzwerk
G.add_node(2)  # entspricht b
G.add_node(3)  # entspricht c
G.add_node(4)  # entspricht d
G.add_node(5)  # entspricht e
G.add_node(6)  # entspricht f

# Kanten hinzufügen
# Hin- und Rückkanten für jedes Kantenpaar
G.add_edge(2, 1, key=0, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(1, 2, key=9, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(1, 2, key=19, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)

G.add_edge(3, 2, key=1, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(2, 3, key=10, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)

G.add_edge(1, 4, key=2, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(4, 1, key=11, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)

G.add_edge(2, 4, key=3, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(4, 2, key=12, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)

G.add_edge(5, 2, key=4, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(2, 5, key=13, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)

G.add_edge(5, 3, key=5, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(3, 5, key=14, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)

G.add_edge(5, 4, key=6, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(4, 5, key=15, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)

G.add_edge(6, 4, key=7, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(4, 6, key=16, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)

G.add_edge(6, 5, key=8, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)
G.add_edge(5, 6, key=17, length=5000, lanes=1, velocity=100, maxspeed=100, PCI=100, time=60, maintenance='no', age=0, duration=0)

nx.draw(G, pos=nx.spring_layout(G))
plt.show()


# Saving
nx.write_gexf(G, "networks_of_investigation/simple_test_graphs/simple_test_graph_BA.gexf")