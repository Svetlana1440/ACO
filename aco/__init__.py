import matplotlib.pyplot as plt
import numpy as np
from ant import Ant
import random

class AntColony:
    def __init__(
        self,
        graph,  # Объект класса Graph
        start_node=0,
        ant_count=10,
        alpha=1.0,
        beta=2.0,
        pheromone_evaporation_rate=0.3,
        pheromone_constant=1.0,
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

        self.num_nodes = len(graph.nodes)
        self.pheromone_map = {
            (i, j): 1 for i in graph.nodes for j in graph.get_neighbors(i)
        }

        self.tmp_pheromone_map = {
            (i, j): 0 for i in graph.nodes for j in graph.get_neighbors(i)
        }

        self.best_path = None
        self.best_cost = float('inf')
        self.path_lengths = [float('inf')] * self.iterations

    def run(self):
        ants = [Ant(alpha=self.alpha, beta=self.beta) for _ in range(self.ant_count)]

        for iteration in range(self.iterations):
            iteration_best_cost = float('inf')
            successful_paths = 0

            for ant in ants:
                ant.reset(self.start_node)
                success, path_cost = self.construct_solution(ant)

                if success:
                    successful_paths += 1
                    if path_cost < self.best_cost:
                        iteration_best_cost = path_cost
                        self.best_cost = iteration_best_cost
                        self.best_path = ant.route

            self.path_lengths[iteration] = self.best_cost
            self.update_pheromones()

        return self.best_path, self.best_cost

    def construct_solution(self, ant):
        current_node = self.start_node

        while len(ant.visited) < self.num_nodes:
            next_node = self.choose_next_node(ant, current_node)
            if next_node is None:
                return False, float('inf')

            ant.route.append(next_node)
            ant.visited.add(next_node)
            current_node = next_node

        if self.graph.get_weight(current_node, self.start_node) < float('inf'):
            ant.route.append(self.start_node)
            path_cost = self.calculate_path_cost(ant.route)
            self.update_tmp_pheromones(ant.route, path_cost)
            return True, path_cost

        return False, float('inf')

    def choose_next_node(self, ant, current_node):
        probabilities = []
        total_probability = 0

        for neighbor, weight in self.graph.get_neighbors(current_node).items():
            if neighbor in ant.visited:
                continue

            pheromone = self.pheromone_map[(current_node, neighbor)]
            heuristic = 1 / weight
            probability = (pheromone ** self.alpha) * (heuristic ** self.beta)

            probabilities.append((neighbor, probability))
            total_probability += probability

        if not probabilities:
            return None

        random_choice = random.uniform(0, total_probability)
        cumulative_probability = 0
        for neighbor, probability in probabilities:
            cumulative_probability += probability
            if cumulative_probability >= random_choice:
                return neighbor

        return None

    def calculate_path_cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            cost += self.graph.get_weight(path[i], path[i + 1])
        return cost

    def update_tmp_pheromones(self, path, path_cost):
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            self.tmp_pheromone_map[(a, b)] += self.pheromone_constant / path_cost
            self.tmp_pheromone_map[(b, a)] += self.pheromone_constant / path_cost

    def update_pheromones(self):
        for edge in self.pheromone_map:
            self.pheromone_map[edge] *= (1 - self.pheromone_evaporation_rate)
            self.pheromone_map[edge] += self.tmp_pheromone_map[edge]
            self.tmp_pheromone_map[edge] = 0

    def plot_graphs(self):
        path_lengths_no_inf = [np.nan if length == float('inf') else length for length in self.path_lengths]
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, self.iterations + 1), path_lengths_no_inf, color='orange')
        plt.title('Path Lengths Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Path Length')
        plt.tight_layout()
        plt.grid()
        plt.show()

