from node import Node

class Graph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node_id):
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(node_id)

    def add_edge(self, node1_id, node2_id, weight):
        self.add_node(node1_id)
        self.add_node(node2_id)
        self.nodes[node1_id].add_edge(node2_id, weight)
        self.nodes[node2_id].add_edge(node1_id, weight)

    def get_neighbors(self, node_id):
        return self.nodes[node_id].edges

    def get_weight(self, node1_id, node2_id):
        return self.nodes[node1_id].edges.get(node2_id, float('inf'))
