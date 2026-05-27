# Class for initialising a qubit node
class Node:

    def __init__(self, name):
        # Name for qubit
        self.name = name
        # Energy of qubit
        self.energy = 1.0
        # Phase of qubit
        self.phase = 0.0
        # Qubit neighbours
        self.neighbours = []
        # Weights to each neighbor
        self.weights = {}
        # Bias for NN forward propagation
        self.bias = 0.0
