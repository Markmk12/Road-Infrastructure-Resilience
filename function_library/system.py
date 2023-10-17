import numpy as np
import networkx as nx


# Network Efficiency (not normalized!)
def network_efficiency(graph):
    n = nx.Graph.number_of_nodes(graph)
    shortest_path_matrix = nx.floyd_warshall_numpy(graph, weight='time')
    efficiency = 1/(n*(n-1))*np.sum(np.divide(1, shortest_path_matrix, where=shortest_path_matrix != 0))
    return efficiency


# def network_efficiency(graph):
#     n = nx.Graph.number_of_nodes(graph)
#     sum_of_inverse_distances = 0.0
#
#     for node in graph.nodes():
#         # Berechnung der kürzesten Pfade von 'node' zu allen anderen Knoten
#         lengths = nx.single_source_dijkstra_path_length(graph, node, weight='time')
#
#         # Addiere die inversen der kürzesten Pfade (ignoriere 0, da dies der Abstand zum Knoten selbst ist)
#         sum_of_inverse_distances += sum(1.0 / distance for distance in lengths.values() if distance != 0)
#
#     efficiency = sum_of_inverse_distances / (n * (n - 1))
#     return efficiency


# def network_efficiency(G):
#     # Berechne die kürzesten Weglängen zwischen allen Paaren von Knoten
#     shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))
#
#     # Berechne die umgekehrten kürzesten Weglängen
#     inv_lengths = [1 / d for source, targets in shortest_path_lengths.items() for target, d in targets.items() if d > 0]
#
#     # Nehme den Durchschnitt der umgekehrten kürzesten Weglängen
#     return np.mean(inv_lengths)


def resilience_metric(q, target_q, observation_time):                   # according Ouyang et al.

    target_q = 1
    resilience = 1 - (target_q * observation_time - np.sum(q)) / (target_q * observation_time)

    return resilience
