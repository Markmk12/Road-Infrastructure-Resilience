import numpy as np
import networkx as nx


# Network Efficiency (not normalized!)
def network_efficiency(graph):
    n = nx.Graph.number_of_nodes(graph)
    shortest_path_matrix = nx.floyd_warshall_numpy(graph)
    efficiency = 1/(n*(n-1))*np.sum(shortest_path_matrix)
    return efficiency
