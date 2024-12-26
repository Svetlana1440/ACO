import matplotlib.pyplot as plt
import random

from aco import AntColony


plt.style.use("dark_background")
## Матрица смежности графа
graph = [
    [0, 2, 2, 1],
    [2, 0, 1, 2],
    [2, 1, 0, 2],
    [1, 2, 2, 0]
]

colony = AntColony(graph, start_node=0, ant_count=10, iterations=100)
best_path, best_cost = colony.run()

print(f"Best path: {best_path}")
print(f"Best cost: {best_cost}")
