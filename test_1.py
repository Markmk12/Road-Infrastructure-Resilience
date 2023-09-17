import networkx as nx
import random
import matplotlib.pyplot as plt


# Funktion zur Durchführung einer MCMC-Simulation
def mcmc_simulation(graph, N):
    # Wörterbuch zur Speicherung des PCI-Verlaufs für jede Kante
    pci_history = {edge: [] for edge in graph.edges()}

    for _ in range(N):
        for edge in graph.edges():
            # Aktuellen PCI-Wert speichern
            pci_history[edge].append(graph[edge[0]][edge[1]]['PCI'])

            # PCI-Wert der Kante verringern
            decrease = random.randint(1, 2)
            graph[edge[0]][edge[1]]['PCI'] -= decrease

            # Sicherstellen, dass der PCI-Wert nicht negativ wird
            if graph[edge[0]][edge[1]]['PCI'] < 0:
                graph[edge[0]][edge[1]]['PCI'] = 0

    return pci_history


# Graph erstellen
G = nx.Graph()
G.add_nodes_from([1, 2, 3, 4])
edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)]
for edge in edges:
    G.add_edge(edge[0], edge[1], PCI=100)

# MCMC-Simulation durchführen
N = 50
pci_history = mcmc_simulation(G, N)

# Abbildung erstellen
plt.figure(figsize=(10, 6))
for edge, history in pci_history.items():
    plt.plot(history, label=f"Kante {edge}")

plt.title('PCI-Verlauf über die Jahre')
plt.xlabel('Jahr')
plt.ylabel('PCI')
plt.legend()
plt.grid(True)
plt.show()
