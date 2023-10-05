import numpy as np
import networkx as nx


# Network Efficiency (not normalized!)
def network_efficiency(graph):
    n = nx.Graph.number_of_nodes(graph)
    shortest_path_matrix = nx.floyd_warshall_numpy(graph, weight='time')
    efficiency = 1/(n*(n-1))*np.sum(np.divide(1, shortest_path_matrix, where=shortest_path_matrix != 0))
    return efficiency


def resilience_metric(q, target_q, observation_time):                   # according Ouyang et al.

    target_q = 1
    resilience = 1 - (target_q * observation_time - np.sum(q)) / (target_q * observation_time)

    return resilience
