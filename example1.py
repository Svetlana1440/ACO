import matplotlib.pyplot as plt
import random

from aco import AntColony


plt.style.use("dark_background")

# Матрица смежности графа
graph = [
    [0, 2, 0, 9, 0, 0, 0, 0, 3, 0],
    [2, 0, 3, 0, 7, 0, 0, 0, 0, 0],
    [0, 3, 0, 4, 0, 8, 0, 0, 0, 0],
    [9, 0, 4, 0, 2, 0, 3, 0, 0, 0],
    [0, 7, 0, 2, 0, 5, 0, 0, 0, 0],
    [0, 0, 8, 0, 5, 0, 3, 6, 0, 0],
    [0, 0, 0, 3, 0, 3, 0, 2, 0, 1],
    [0, 0, 0, 0, 0, 6, 2, 0, 0, 4],
    [3, 0, 0, 0, 0, 0, 0, 0, 0, 7],
    [0, 0, 0, 0, 0, 0, 1, 4, 7, 0]
]


colony = AntColony(graph, start_node=0, ant_count=10, iterations=100)
best_path, best_cost = colony.run()

print(f"Best path: {best_path}")
print(f"Best cost: {best_cost}")

# Построение графиков
colony.plot_graphs()