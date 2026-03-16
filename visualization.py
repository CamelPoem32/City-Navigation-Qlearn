import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def visualize_graph(G, V, policy, nodes, node_index, turns, node_size=250, show_node_id=False, arrowwidth=2.5, show_trajectory_only=False, start=0):

    pos = {n:G.nodes[n]["pos"] for n in G.nodes}

    values = np.array([V[node_index[n]] for n in nodes])

    fig, ax = plt.subplots(figsize=(14,10))

    # draw nodes colored by value
    node_plot = nx.draw_networkx_nodes(
        G,
        pos,
        node_color=values,
        cmap="plasma",
        node_size=node_size,
        ax=ax
    )

# draw normal edges
    nx.draw_networkx_edges(
        G,
        pos,
        alpha=0.3,
        arrows=True,
        ax=ax
    )

    # draw policy arrows (red)
    policy_edges=[]
    policy_turns=[]

    for s,a in enumerate(policy):

        if a is None:
            continue

        u = nodes[s]
        v = nodes[a]

        policy_edges.append((u,v))

    if show_trajectory_only:

        traj=[]
        s=start

        visited=set()

        while s is not None and s not in visited:

            visited.add(s)

            a=policy[s]

            if a is None:
                break

            next_s = nodes[a]

            traj.append((nodes[s], next_s))

            if (policy[a] is not None) and (not np.isnan(policy[a])): 
                w = nodes[policy[a]]
                if (w is not None) and (not np.isnan(policy[a])): 
                    policy_turns.append((nodes[s],nodes[a],w))

            s=a

        nx.draw_networkx_edges(G,pos,edgelist=traj,
                            edge_color="red",
                            width=3,
                            arrows=True,
                            ax=ax)

    else:

        nx.draw_networkx_edges(G,pos,
                            edgelist=policy_edges,
                            edge_color="red",
                            width=2.5,
                            arrows=True,
                            ax=ax)

    # draw node labels
    if show_node_id:
        nx.draw_networkx_labels(
            G,
            pos,
            font_size=7,
            ax=ax
        )

    # draw turn symbols
    for (u,v,w) in policy_turns:
        if u==v or u==w or v==w: continue
        t = turns[(u,v,w)]

        p1 = pos[u]
        p2 = pos[v]
        p3 = pos[w]

        mid = (p1 + p2) / 2

        if t=="left":
            symbol="←"
            color="blue"

        elif t=="right":
            symbol="→"
            color="green"

        else:
            symbol="↑"
            color="gray"

        ax.text(
            mid[0],
            mid[1],
            symbol,
            color=color,
            fontsize=20,
            ha="center",
            va="center"
        )

    # proper colorbar
    fig.colorbar(node_plot, ax=ax, label="Value")

    ax.set_title("Policy + Value Function + Turn Detection")

    ax.axis("off")

    plt.show()

    return fig

def plot_convergence(V_history, Q_history, conv, action_names=None):

    max_v = np.max([len(V_history[i]) for i in range(len(V_history))])
    n_q = np.max([len(Q_history[i]) for i in range(len(Q_history))])
    # max_q = np.max([len(Q_history[i]) for i in range(len(Q_history))])
    V_arr = np.empty((len(V_history), max_v))
    for i, v in enumerate(V_history):
        V_arr[i][:len(v)] = v
        V_arr[i][len(v):] = v[-1]
    Q_arr = np.empty((len(Q_history), n_q, max_v))
    for i, q in enumerate(Q_history):
        for j in range(len(q)):
            Q_arr[i, j][:len(q[j])] = q[j]
            Q_arr[i, j][len(q[j]):] = q[j][-1]

    iters = np.arange(len(V_arr))

    NUM_STATES = V_arr.shape[1]
    NUM_ACTIONS = Q_arr.shape[2]

    fig, axes = plt.subplots(1,3,figsize=(18,5))

    # V convergence
    ax=axes[0]

    colors = plt.cm.tab10(np.linspace(0,1,min(NUM_STATES, 25)))

    for s in range(min(NUM_STATES, 25)):
        ax.plot(iters,V_arr[:,s],color=colors[s],linewidth=2,label=f"s={s}")

    ax.set_title("V(s) convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("V(s)")
    ax.grid(True)
    ax.legend(fontsize=7)

    # Q convergence
    ax=axes[1]

    linestyles=['-','--',':','-.']

    for s in range(min(NUM_STATES, 25)):

        for a in range(NUM_ACTIONS):

            ax.plot(iters,Q_arr[:,s,a],
                    color=colors[s],
                    linestyle=linestyles[a%4],
                    alpha=0.7)

    ax.set_title("Q(s,a) convergence")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Q(s,a)")
    ax.grid(True)

    # convergence metric
    ax=axes[2]

    ax.semilogy(conv,color="crimson",linewidth=2)

    ax.set_title("max update per iteration")
    ax.set_xlabel("iteration")
    ax.set_ylabel("||Δ||∞")

    ax.grid(True)

    plt.tight_layout()
    plt.show()

    return fig