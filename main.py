# Importing graph and node
from Node import Node
from Graph import Graph
import math, random

# Instantiating graph
graph = Graph()

from graph_representation import draw_graph, plot_loss, plot_phase_evolution

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

# Initialise phases
Q1.phase = 0.0
Q2.phase = 1.2
Q3.phase = 2.1
Q4.phase = 0.8

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

# Other variables
Learning_rate = 0.001
Losses = []

#Math functions
'''Used to ensure that values being passed in the hidden layer are 0 or higher, no negative numbers (less than zero)'''
def Relu(x):
    return max(0, x)

'''Triggers in the output layer of the neural network, it then takes the final values and runs the 'sigmoid' function
which is responsible for calculating the probability of the network ai thinking it is correct (AI confidemce from 0-1)'''
def Sigmoid(z):
    return 1 / (1 + (math.exp(-z) ) )

'''This is forward propagation using neurons in a graph format instead of a forward feeding format'''
def forward_propagate(graph):

    new_energies = {}

    # Loop through each node in graph
    for node in graph.nodes:

        # Variable used to store value of neuron propagation
        z = 0

        # Looping through neighbour weights
        for neighbour in node.neighbours:
            weight = node.weights[neighbour]
            
            # Neuron calculation (sum(a*b)+bias) with added use of phases
            phase_difference = node.phase - neighbour.phase
            interference = math.cos(phase_difference)
            z += neighbour.energy * weight * interference

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
        neighbour_total = 0

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

'''Calculates a loss by using MSE (Mean Square Error), mainly used for developer debugging'''
def calculate_loss(y_pred, y_true):
    loss = 0
    # Loops through list of predicted and true values and runs MSE calculation
    for pred, true in zip(y_pred, y_true):
        loss += (pred - true) ** 2

    return (loss / len(y_pred))



'''This is back propagation where we will be calculating a loss and comparing to the AI's results and altering the according weights and biases to improve the model'''
def back_propagate(graph, y_pred, y_true):
    # Assigns each node in teh graph an index and loops through then to calculate a loss
    for node_index, node in enumerate(graph.nodes):
        error = y_pred[node_index] - y_true[node_index]

        # Multiply loss (error) by each neighbour energy (calculates a gradient to see how much the mistake contributes)
        for neighbour in node.neighbours:
            gradient = error * neighbour.energy
            node.weights[neighbour] -= (Learning_rate * gradient)
        # Overwrites biases with the new 'learned' values
        node.bias -= (Learning_rate * error)



phase_history = {
    "Q1": [],
    "Q2": [],
    "Q3": [],
    "Q4": []
}

'''------ Main Loop option 1 ------'''
def Single_step_training():
    # 4 full forward and back prop loops
    for i in range(4):

        # Calculates what values should be before forward prop even runs
        y_true = generate_true_future(graph)
        
        print(f"------ LOOP {i+1} ------")
        print()

        # Used for debugging
        print('normal value before any change')
        # Prints all details about each node
        for node in graph.nodes:
            print("Name:", node.name, "Energy:", node.energy, "Bias:", node.bias, "Phase:", node.phase)
            # Prints each neighbour of the node and their weights
            temp = []
            for neighbour, weight in node.weights.items():
                temp.append(f"{node.name} -> {neighbour.name} : {weight}")
            print("Neighbours and weights:",temp)
            print()

        # Forward propagation and reandomly increments or decrements phase after propagation to model unstable qubits
        forward_propagate(graph)
        for node in graph.nodes:
            node.phase += random.uniform(-0.05,0.05)

        for node in graph.nodes:
            phase_history[node.name].append(node.phase)

        # Used for debugging
        print('after forward prop')
        # Prints all details about each node
        print('normal value before any change')
        for node in graph.nodes:
            print("Name:", node.name, "Energy:", node.energy, "Bias:", node.bias, "Phase:", node.phase)
            # Prints each neighbour of the node and their weights
            temp = []
            for neighbour, weight in node.weights.items():
                temp.append(f"{node.name} -> {neighbour.name} : {weight}")
            print("Neighbours and weights:",temp)
            print()

        # Gets prediction of graph after forward prop to compare with 'true values' from before forward prop
        y_pred = get_predictions(graph)
        loss = calculate_loss(y_pred, y_true)
        Losses.append(loss)

        # Used for debugging (NO LONGER NEEDED)
        #print('Loss calculation')
        #print(loss)

        back_propagate(graph, y_pred, y_true)

        # Used for debugging
        print('after back prop')
        print('normal value before any change')
        # Prints all details about each node
        for node in graph.nodes:
            print("Name:", node.name, "Energy:", node.energy, "Bias:", node.bias, "Phase:", node.phase)
            # Prints each neighbour of the node and their weights
            temp = []
            for neighbour, weight in node.weights.items():
                temp.append(f"{node.name} -> {neighbour.name} : {weight}")
            print("Neighbours and weights:",temp)
            print()
        

    #(NO LONGER NEEDED)
    #for node in graph.nodes:
    #    print(node.name, node.energy, node.bias, node.weights)



'''------ Main Loop option 2 ------'''
def Multi_step_training():
    for i in range (4):
        y_true = generate_true_future(graph)
        for i in range(4):
            forward_propagate(graph)
        y_pred = get_predictions(graph)
        loss = calculate_loss(y_pred, y_true)
        back_propagate(graph, y_pred, y_true)

        for node in graph.nodes:
            print(node.name, node.energy)


# Just using single step training for now, I have no intent of using multi step training as of yet
Single_step_training()
# Prints out losses in list format
print("If loss trends downwards then learning has been demonstrated:")
for index, loss in enumerate(Losses):
    print(f"Loss {index + 1}:",loss)

print()

# Prints sigmoid version and none sigmoid version of qubits results
for node in graph.nodes:
    confidence = Sigmoid(node.energy)
    print(node.name, "Energy:", node.energy)
    print(node.name, "Confidence:", confidence)
    print()

draw_graph(graph)
plot_loss(Losses)
plot_phase_evolution(phase_history)