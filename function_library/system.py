import numpy as np
import networkx as nx


# Network Efficiency (not normalized)
def network_efficiency(graph):
    """
    Calculate the non-standardized efficiency of a given graph.

    The efficiency of a graph is defined as the average of the inverse shortest
    path lengths between all pairs of nodes. It represents a measure of the average
    accessibility between nodes in a network. For the calculation the proposed formula of
    Latora and Marchiori (2001) is used.

    Parameters:
    -----------
    graph : networkx.Graph or networkx.MultiDiGraph
        The graph (or MultiDiGraph) for which the global efficiency is to be calculated.
        The edges of the graph should have a 'time' attribute, which will be used as weight
        for shortest paths.

    Returns:
    --------
    float
        The global efficiency of the given graph. This value ranges between 0 and 1,
        with 1 representing the highest possible efficiency.

    Raises:
    -------
    ValueError
        If edges without the 'time' attribute are found in the graph.
    Warning
        If NaN values are detected in the shortest path matrix.

    """

    n = nx.Graph.number_of_nodes(graph)
    shortest_path_matrix = nx.floyd_warshall_numpy(graph, weight='time')

    # Check for NaN values in shortest_path_matrix
    if np.any(np.isnan(shortest_path_matrix)):
        print("NaN values detected in shortest path matrix!")

    # Prevent division by zero. This occurs especially when there are zeros on the main diagonal of
    # the shortest_path_matrix (node i=j).
    inverse_distances = np.divide(1, shortest_path_matrix, out=np.zeros_like(shortest_path_matrix), where=shortest_path_matrix != 0)
    efficiency = 1 / (n * (n - 1)) * np.sum(inverse_distances)

    return efficiency


# Resilience metric
def resilience_metric(q, observation_time):
    """
    Calculate the resilience of a system using the probabilistic resilience metric of Ouyang et al. (2012).

    The resilience metric takes values between 0 and 1. For simplicity the target efficiency is a non-random constant
    and the rewritten formula of Salomon et al. (2020) is implemented.



    Parameters:
    -----------
    q : array
        An array of normalized system efficiencies observed over time.
        Each value in `q` should be between 0 and 1, with 1 representing optimal efficiency.

    observation_time : int or float
        The total duration of the observation period.

    Returns:
    --------
    float
        The resilience metric of the system. This value ranges between 0 and 1,
        with 1 representing the highest possible resilience.

    """

    target_q = 1
    resilience = 1 - (target_q * observation_time - np.sum(q)) / (target_q * observation_time)

    return resilience
