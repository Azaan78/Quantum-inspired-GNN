import matplotlib.pyplot as plt
import networkx as nx


def visualise_project(initial_graph,
                      final_graph,
                      losses,
                      phase_history):

    # ==================================================
    # Create 2x2 dashboard
    # ==================================================

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax_initial = axes[0, 0]
    ax_final = axes[0, 1]
    ax_loss = axes[1, 0]
    ax_phase = axes[1, 1]

    # ==================================================
    # INITIAL GRAPH
    # ==================================================

    G_initial = nx.Graph()

    for node in initial_graph.nodes:
        G_initial.add_node(
            node.name,
            energy=node.energy,
            phase=node.phase
        )

    for node in initial_graph.nodes:
        for neighbour, weight in node.weights.items():

            if not G_initial.has_edge(
                    node.name,
                    neighbour.name):

                G_initial.add_edge(
                    node.name,
                    neighbour.name,
                    weight=weight
                )

    pos_initial = nx.spring_layout(
        G_initial,
        seed=42
    )

    node_sizes = [
        max(
            200,
            G_initial.nodes[n]["energy"] * 300
        )
        for n in G_initial.nodes
    ]

    phases = [
        G_initial.nodes[n]["phase"]
        for n in G_initial.nodes
    ]

    nodes_initial = nx.draw_networkx_nodes(
        G_initial,
        pos_initial,
        node_size=node_sizes,
        node_color=phases,
        cmap=plt.cm.plasma,
        ax=ax_initial
    )

    for u, v, d in G_initial.edges(data=True):

        weight = d["weight"]

        nx.draw_networkx_edges(
            G_initial,
            pos_initial,
            edgelist=[(u, v)],
            width=max(
                0.5,
                abs(weight) * 2
            ),
            edge_color=(
                "green"
                if weight > 0
                else "red"
            ),
            alpha=0.7,
            ax=ax_initial
        )

    nx.draw_networkx_labels(
        G_initial,
        pos_initial,
        ax=ax_initial
    )

    ax_initial.set_title(
        "Initial Quantum Graph"
    )

    ax_initial.axis("off")

    # ==================================================
    # FINAL GRAPH
    # ==================================================

    G_final = nx.Graph()

    for node in final_graph.nodes:
        G_final.add_node(
            node.name,
            energy=node.energy,
            phase=node.phase
        )

    for node in final_graph.nodes:
        for neighbour, weight in node.weights.items():

            if not G_final.has_edge(
                    node.name,
                    neighbour.name):

                G_final.add_edge(
                    node.name,
                    neighbour.name,
                    weight=weight
                )

    pos_final = nx.spring_layout(
        G_final,
        seed=42
    )

    node_sizes = [
        max(
            200,
            G_final.nodes[n]["energy"] * 300
        )
        for n in G_final.nodes
    ]

    phases = [
        G_final.nodes[n]["phase"]
        for n in G_final.nodes
    ]

    nodes_final = nx.draw_networkx_nodes(
        G_final,
        pos_final,
        node_size=node_sizes,
        node_color=phases,
        cmap=plt.cm.plasma,
        ax=ax_final
    )

    for u, v, d in G_final.edges(data=True):

        weight = d["weight"]

        nx.draw_networkx_edges(
            G_final,
            pos_final,
            edgelist=[(u, v)],
            width=max(
                0.5,
                abs(weight) * 2
            ),
            edge_color=(
                "green"
                if weight > 0
                else "red"
            ),
            alpha=0.7,
            ax=ax_final
        )

    nx.draw_networkx_labels(
        G_final,
        pos_final,
        ax=ax_final
    )

    ax_final.set_title(
        "Final Quantum Graph"
    )

    ax_final.axis("off")

    # ==================================================
    # LOSS CURVE
    # ==================================================

    ax_loss.plot(
        losses,
        linewidth=2
    )

    ax_loss.set_title(
        "Training Loss"
    )

    ax_loss.set_xlabel(
        "Epoch"
    )

    ax_loss.set_ylabel(
        "MSE Loss"
    )

    ax_loss.grid(True)

    # ==================================================
    # PHASE EVOLUTION
    # ==================================================

    for node_name, phases in phase_history.items():

        ax_phase.plot(
            phases,
            label=node_name
        )

    ax_phase.set_title(
        "Phase Evolution"
    )

    ax_phase.set_xlabel(
        "Epoch"
    )

    ax_phase.set_ylabel(
        "Phase"
    )

    ax_phase.legend()

    ax_phase.grid(True)

    # ==================================================
    # Colourbar
    # ==================================================

    fig.colorbar(
        nodes_final,
        ax=[ax_initial, ax_final],
        shrink=0.8,
        label="Phase"
    )

    plt.tight_layout()

    plt.show()