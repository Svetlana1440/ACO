import matplotlib.pyplot as plt
from aco import AntColony


# Матрица смежности графа
graph = [
    [0, 3, 0, 0, 1, 0],  # вершина 0
    [3, 0, 18, 0, 0, 3], # вершина 1
    [0, 3, 0, 1, 0, 1],  # вершина 2
    [0, 0, 8, 0, 1, 5],  # вершина 3
    [3, 0, 0, 3, 0, 4],  # вершина 4
    [3, 3, 3, 5, 4, 0]   # вершина 5
]



colony = AntColony(graph, start_node=0, ant_count=1, iterations=100, alpha=2, beta=1, pheromone_evaporation_rate=0.3, pheromone_constant=0.5)
best_path, best_cost = colony.run()

print(f"Best path: {best_path}")
print(f"Best cost: {best_cost}")

# Построение графиков
colony.plot_graphs()