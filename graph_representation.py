import matplotlib.pyplot as plt
import networkx as nx

# ----------------------------
# GRAPH VISUALISATION
# ----------------------------

def draw_graph(graph):
    G = nx.Graph()

    # ----------------------------
    # Build NetworkX graph
    # ----------------------------
    for node in graph.nodes:
        G.add_node(node.name,
                   energy=node.energy,
                   phase=node.phase)

    for node in graph.nodes:
        for neighbour, weight in node.weights.items():
            if not G.has_edge(node.name, neighbour.name):
                G.add_edge(node.name,
                           neighbour.name,
                           weight=weight)

    pos = nx.spring_layout(G, seed=42)

    # ----------------------------
    # Node styling
    # ----------------------------
    node_sizes = [max(50, G.nodes[n]['energy'] * 300) for n in G.nodes]
    phases = [G.nodes[n]['phase'] for n in G.nodes]

    # ----------------------------
    # Figure + Axes (IMPORTANT FIX)
    # ----------------------------
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw nodes (THIS is the mappable for colorbar)
    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_size=node_sizes,
        node_color=phases,
        cmap=plt.cm.plasma,
        ax=ax
    )

    # Draw edges
    for (u, v, d) in G.edges(data=True):
        weight = d['weight']

        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=max(0.5, abs(weight) * 2),
            edge_color='green' if weight > 0 else 'red',
            alpha=0.6,
            ax=ax
        )

    # Labels
    nx.draw_networkx_labels(G, pos, ax=ax)

    # ----------------------------
    # FIXED COLORBAR (KEY PART)
    # ----------------------------
    cbar = fig.colorbar(nodes, ax=ax)
    cbar.set_label("Phase")

    # ----------------------------
    # Final styling
    # ----------------------------
    ax.set_title("Quantum-Inspired GNN Graph State")
    ax.axis("off")

    plt.show()


# ----------------------------
# LOSS CURVE
# ----------------------------

def plot_loss(losses):
    plt.figure()
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.show()


# ----------------------------
# PHASE EVOLUTION
# ----------------------------

def plot_phase_evolution(phase_history):
    plt.figure()

    for node_name, phases in phase_history.items():
        plt.plot(phases, label=node_name)

    plt.title("Phase Evolution Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Phase")
    plt.legend()
    plt.show()