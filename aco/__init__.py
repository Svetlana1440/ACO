import random
import matplotlib.pyplot as plt

class Ant:
    def __init__(
        self,
        graph,  # Матрица смежности
        pheromone_map,
        tmp_pheromone_map,
        start_node,
        alpha,
        beta,
        pheromone_constant,
    ):
        self.graph = graph
        self.pheromone_map = pheromone_map
        self.tmp_pheromone_map = tmp_pheromone_map
        self.alpha = alpha
        self.beta = beta
        self.pheromone_constant = pheromone_constant
        self.start_node = start_node

        self.num_nodes = len(graph)
        self.current_node = start_node
        self.visited = set()
        self.path = []
        self.path_cost = 0

    def run(self):
        self.path = [self.start_node]
        self.visited.add(self.start_node)
        self.current_node = self.start_node

        while len(self.visited) < self.num_nodes:
            next_node = self.choose_next_node()
            if next_node is None:
                # Если муравей "застрял", прерываем его путь
                return False

            self.path_cost += self.graph[self.current_node][next_node]
            self.path.append(next_node)
            self.visited.add(next_node)
            self.current_node = next_node

        # Замыкаем цикл, возвращаясь в начальную вершину
        if self.graph[self.current_node][self.start_node] > 0:
            self.path_cost += self.graph[self.current_node][self.start_node]
            self.path.append(self.start_node)
            self.update_pheromones()
            return True

        return False  # Если цикл не удалось завершить

    def choose_next_node(self):
        probabilities = []
        total_probability = 0

        for neighbor, distance in enumerate(self.graph[self.current_node]):
            if neighbor in self.visited or distance == 0:
                continue

            pheromone = self.pheromone_map[self.current_node][neighbor]
            heuristic = 1 / distance
            probability = (pheromone ** self.alpha) * (heuristic ** self.beta)

            probabilities.append((neighbor, probability))
            total_probability += probability

        if not probabilities:
            return None

        # Рулетка для выбора следующей вершины
        random_choice = random.uniform(0, total_probability)
        cumulative_probability = 0
        for neighbor, probability in probabilities:
            cumulative_probability += probability
            if cumulative_probability >= random_choice:
                return neighbor

        return None

    def update_pheromones(self):
        for i in range(len(self.path) - 1):
            a, b = self.path[i], self.path[i + 1]
            self.tmp_pheromone_map[a][b] += self.pheromone_constant / self.path_cost
            self.tmp_pheromone_map[b][a] += self.pheromone_constant / self.path_cost


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
        self.path_lengths = []
        self.best_path_probabilities = []

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
                    if ant.path_cost < iteration_best_cost:
                        iteration_best_cost = ant.path_cost
                        self.best_path = ant.path

            if self.best_path:
                self.best_cost = iteration_best_cost

            # Запись длины пути и вероятности по лучшему пути
            self.path_lengths.append(self.best_cost)
            if successful_paths > 0:
                best_path_probability = successful_paths / self.ant_count
            else:
                best_path_probability = 0
            self.best_path_probabilities.append(best_path_probability)

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
        # График изменения длины пути на каждой итерации
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.iterations + 1), self.path_lengths, marker='o', color='b')
        plt.title('Path Lengths Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Path Length')

        # График изменения вероятности прохождения по лучшему пути
        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.iterations + 1), self.best_path_probabilities, marker='o', color='r')
        plt.title('Best Path Probability Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Best Path Probability')

        plt.tight_layout()
        plt.show()
