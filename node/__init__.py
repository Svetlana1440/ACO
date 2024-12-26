class Node:
    def __init__(self, id):
        self.id = id
        self.edges = {}  # Соседние узлы и их веса

    def add_edge(self, neighbor, weight):
        self.edges[neighbor] = weight
