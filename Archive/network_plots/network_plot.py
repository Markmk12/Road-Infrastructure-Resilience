import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_gexf("hamburg.gexf")

# Plotting
nx.draw(G, with_labels=True)
plt.show()