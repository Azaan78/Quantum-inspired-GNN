# Class for creating graph of nodes
class Graph:

    # Initialises all nodes in the graph
    def __init__(self):
        self.nodes = []

    # Adds a node to the graph
    def add_node(self, node):
        self.nodes.append(node)

    # Adds a link to the graph
    def connect_nodes(self, node1, node2):
        # Graph adds a bi-directional edge
        node1.neighbours.append(node2)
        node2.neighbours.append(node1)
