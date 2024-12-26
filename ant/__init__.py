

class Ant:
    def __init__(self, alpha=1, beta=1):
        self.alpha = alpha
        self.beta = beta
        self.route = []
        self.visited = set()

    def reset(self, start_node):
        self.route = [start_node]
        self.visited = {start_node}
