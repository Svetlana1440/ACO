import matplotlib.pyplot as plt
import numpy as np
from ant import Ant



class AntColony:
    def __init__(
        self,
        graph,  # Матрица смежности
        start_node=0,
        ant_count=10,
        alpha=1.0,
        beta=2.0,
        pheromone_evaporation_rate=0.1,
        pheromone_constant=100.0,
        iterations=100,
    ):
        self.graph = graph
        self.start_node = start_node
        self.ant_count = ant_count
        self.alpha = alpha
        self.beta = beta
        self.pheromone_evaporation_rate = pheromone_evaporation_rate
        self.pheromone_constant = pheromone_constant
        self.iterations = iterations

        self.num_nodes = len(graph)
        self.pheromone_map = [[1 for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]
        self.tmp_pheromone_map = [[0 for _ in range(self.num_nodes)] for _ in range(self.num_nodes)]

        self.best_path = None
        self.best_cost = float('inf')

        # Для хранения данных для графиков
        self.path_lengths = [float('inf')] * self.iterations  # Инициализируем список с бесконечностью

    def run(self):
        for iteration in range(self.iterations):
            ants = [Ant(
                self.graph,
                self.pheromone_map,
                self.tmp_pheromone_map,
                self.start_node,
                self.alpha,
                self.beta,
                self.pheromone_constant,
            ) for _ in range(self.ant_count)]

            iteration_best_cost = float('inf')
            successful_paths = 0
            for ant in ants:
                success = ant.run()

                if success:
                    successful_paths += 1
                    if ant.path_cost < self.best_cost:
                        iteration_best_cost = ant.path_cost
                        self.best_cost = iteration_best_cost
                        self.best_path = ant.path

            # Запись длины пути и вероятности по лучшему пути
            self.path_lengths[iteration] = self.best_cost

            self.update_pheromones()

        return self.best_path, self.best_cost

    def update_pheromones(self):
        # Испарение феромонов
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                self.pheromone_map[i][j] *= (1 - self.pheromone_evaporation_rate)
                self.pheromone_map[i][j] += self.tmp_pheromone_map[i][j]
                self.tmp_pheromone_map[i][j] = 0
                
    def plot_graphs(self):
        # Заменим значения inf на NaN, чтобы они не отображались на графике
        path_lengths_no_inf = [np.nan if length == float('inf') else length for length in self.path_lengths]

        # График изменения длины пути на каждой итерации
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, self.iterations + 1), path_lengths_no_inf, color='orange')
        plt.title('Path Lengths Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Path Length')

        plt.tight_layout()
        plt.show()