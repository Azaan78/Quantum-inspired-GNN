# Importing graph and node
from Node import Node
from Graph import Graph
import math

# Instantiating graph
graph = Graph()

# Instantiating qubits
Q1 = Node("Q1")
Q2 = Node("Q2")
Q3 = Node("Q3")
Q4 = Node("Q4")

# Initialising energies
Q1.energy = 1
Q2.energy = 7
Q3.energy = 3
Q4.energy = 10

# Initialisin weights
Q1.weights[Q2] = 0.8
Q1.weights[Q3] = -0.3
Q1.weights[Q4] = 1.2

Q2.weights[Q1] = 0.5
Q2.weights[Q3] = 0.7
Q2.weights[Q4] = -0.4

Q3.weights[Q1] = 1.1
Q3.weights[Q2] = -0.6
Q3.weights[Q4] = 0.9

Q4.weights[Q1] = 0.3
Q4.weights[Q2] = 0.4
Q4.weights[Q3] = 1.5

# Initialise biases
Q1.bias = 0.5
Q2.bias = -0.2
Q3.bias = 0.1
Q4.bias = 0.8

# Add nodes to graph
graph.add_node(Q1)
graph.add_node(Q2)
graph.add_node(Q3)
graph.add_node(Q4)

# Initialising qubit neighbours
graph.connect_nodes(Q1, Q2)
graph.connect_nodes(Q1, Q3)
graph.connect_nodes(Q1, Q4)

graph.connect_nodes(Q2, Q3)
graph.connect_nodes(Q2, Q4)

graph.connect_nodes(Q3, Q4)



#Math functions
'''Used to ensure that values being passed in the hidden layer are 0 or higher, no negative numbers (less than zero)'''
def Relu(x):
    return max(0, x)

'''Triggers in the output layer of the neural network, it then takes the final values and runs the 'sigmoid' function
which is responsible for calculating the probability of the network ai thinking it is correct (AI confidemce from 0-1)'''
def Sigmoid(z):
    return 1 / (1 + (math.exp(-z) ) )

'''This is forward propagation using neurons in a graph format instead of a forward feeding format'''
def propagate(graph):

    new_energies = {}

    # Loop through each node in graph
    for node in graph.nodes:

        # Variable used to store value of neuron propagation
        z = 0

        # Looping through neighbour weights
        for neighbour in node.neighbours:
            print(neighbour)
            weight = node.weights[neighbour]
            
            # Neuron calculation (sum(a*b)+bias)
            z += neighbour.energy * weight

        # Add bias
        z += node.bias

        # Runs relu activation
        new_energy = Relu(z)
        # Saves new node with their energy
        new_energies[node] = new_energy

    # Update energies of all nodes
    for node in graph.nodes:
        node.energy = new_energies[node]



'''This function is used to calculate what the result should be, this is used in loss function later (also known as y_true)'''
def generate_true_future(graph):
    future_states = []

    # Loops through each node in the graph
    for node in graph.nodes:
        neigbour_total = 0

        # Keeps running total of node neighbours
        for neighbour in node.neighbours:
            neighbour_total += neighbour.energy

        # Calculates an average of neighbours energy
        neighbour_average = (neighbour_total / len(node.neighbours))
        new_energy = ((node.energy + neighbour_average) / 2)
        future_states.append(new_energy)
    # Return 'true' states
    return future_states

'''Saves current node energies to list called predictions (also known as y_pred)'''
def get_predictions(graph):
    predictions = []
    for node in graph.nodes:
        predictions.append(node.energy)
    return predictions



'''Calculates a loss by using MSE (Mean Square Error)'''
def calculate_loss(y_pred, y_true):
    loss = 0
    # Loops through list of predicted and tru values and runs MSE calculation
    for pred, true in zip(y_pred, y_true):
        loss += (pred - true) ** 2

    return loss



'''------ Main Loop ------'''
for i in range (4):
    propagate(graph)

    for node in graph.nodes:
        print(node.name, node.energy)

