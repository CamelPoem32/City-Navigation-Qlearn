import numpy as np
import networkx as nx
import osmnx as ox

from settings import *

def create_city_graph(place="Helsinki, Finland"):

    G = ox.graph_from_place(
        place,
        network_type="drive",
        simplify=True
    )

    nodes = list(G.nodes)
    node_index = {n:i for i,n in enumerate(nodes)}

    coords = np.array([
        [G.nodes[n]["x"], G.nodes[n]["y"]]
        for n in nodes
    ])

    return G, nodes, node_index, coords

def classify_turn(p1, p2, p3):

    v1 = p2 - p1
    v2 = p3 - p2

    cross = v1[0]*v2[1] - v1[1]*v2[0]
    dot = v1 @ v2

    angle = np.arctan2(cross, dot)

    if abs(angle) < np.deg2rad(10):
        return "straight"
    elif angle > 0:
        return "left"
    else:
        return "right"
    
def detect_turns(G):

    turns = {}

    for v in G.nodes:

        preds = list(G.predecessors(v))
        succs = list(G.successors(v))

        for u in preds:
            for w in succs:

                if u == w:
                    continue

                p1 = G.nodes[u]["pos"]
                p2 = G.nodes[v]["pos"]
                p3 = G.nodes[w]["pos"]

                turn = classify_turn(p1,p2,p3)

                turns[(u,v,w)] = turn

    return turns
    
def build_mdp(G, nodes, node_index, coords):

    N = len(nodes)

    neighbors = []
    costs = []

    for n in nodes:

        succ = list(G.successors(n))

        idx = [node_index[s] for s in succ]

        c = []

        for s in succ:

            p1 = coords[node_index[n]]
            p2 = coords[node_index[s]]

            dist = np.linalg.norm(p2-p1)

            travel_time = dist/10

            # simplified: treat all as straight
            wait = 40

            cost = travel_time + wait

            c.append(cost)

        neighbors.append(np.array(idx))
        costs.append(np.array(c))

    return neighbors, costs

def build_neighbors_costs(G, nodes, node_index, turns):

    neighbors = []
    costs = []

    for u in nodes:

        succ = list(G.successors(u))

        next_ids = []
        edge_costs = []

        for v in succ:

            next_ids.append(node_index[v])

            # base travel cost
            p1 = G.nodes[u]["pos"]
            p2 = G.nodes[v]["pos"]

            dist = np.linalg.norm(p2 - p1)
            travel_time = dist / SPEED

            # turn penalty (approximate)
            turn_type = "straight"

            preds = list(G.predecessors(u))

            if preds:
                p = preds[0]

                triple = (p, u, v)

                if triple in turns:
                    turn_type = turns[triple]

            if turn_type == "left":
                wait = LEFT_WAIT
            elif turn_type == "right":
                wait = RIGHT_WAIT
            else:
                wait = STRAIGHT_WAIT

            edge_costs.append(travel_time + wait)

        neighbors.append(np.array(next_ids, dtype=int))
        costs.append(np.array(edge_costs))

    return neighbors, costs

def create_artificial_graph(n=4):

    G = nx.DiGraph()

    for i in range(n):
        for j in range(n):
            G.add_node((i,j), pos=np.array([i,j]))

    for i in range(n):
        for j in range(n):

            if i+1 < n:
                G.add_edge((i,j),(i+1,j))
                G.add_edge((i+1,j),(i,j))

            if j+1 < n:
                G.add_edge((i,j),(i,j+1))
                G.add_edge((i,j+1),(i,j))

    return G

def create_city_graph(place="Helsinki, Finland"):

    G = ox.graph_from_place(
        place,
        network_type="drive",
        simplify=True
    )

    # store coordinates as numpy arrays
    for n,data in G.nodes(data=True):
        data["pos"] = np.array([data["x"], data["y"]])

    return G

def graph_to_numpy(G):

    nodes = list(G.nodes)
    node_index = {n:i for i,n in enumerate(nodes)}

    coords = np.array([G.nodes[n]["pos"] for n in nodes])

    neighbors = []

    for n in nodes:

        succ = list(G.successors(n))

        neighbors.append(
            np.array([node_index[s] for s in succ])
        )

    return nodes, node_index, coords, neighbors