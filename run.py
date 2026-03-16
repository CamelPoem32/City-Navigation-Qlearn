import argparse
import os
import numpy as np

from algorithms import (
    value_iteration,
    policy_iteration,
    q_learning,
    q_learning_njit,
    pad_arrays,
)

from env import (
    create_city_graph,
    create_artificial_graph,
    detect_turns,
    graph_to_numpy,
    build_neighbors_costs,
)

from visualization import (
    visualize_graph,
    plot_convergence,
)


def parse_args():

    parser = argparse.ArgumentParser(description="RL Logistics Routing")

    parser.add_argument(
        "--method",
        choices=["value", "policy", "q", "q_njit"],
        required=True,
        help="algorithm to run",
    )

    parser.add_argument(
        "--graph",
        choices=["artificial", "city"],
        default="artificial",
        help="graph type",
    )

    parser.add_argument(
        "--place",
        type=str,
        default="Piedmont, California, USA",
        help="city name for OSM graph",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=5,
        help="size of artificial graph",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=3000,
        help="episodes for Q-learning",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="learning rate",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=0.9,
        help="epsilon exploration",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="directory to save plots",
    )

    return parser.parse_args()


def build_graph(args):

    if args.graph == "artificial":

        G = create_artificial_graph(args.size)

    else:

        G = create_city_graph(args.place)

    turns = detect_turns(G)

    nodes, node_index, coords, neighbors = graph_to_numpy(G)

    neighbors, costs = build_neighbors_costs(G, nodes, node_index, turns)

    return G, nodes, node_index, neighbors, costs, turns


def run_value_iteration(neighbors, costs, gamma):

    V, policy, Vh, Qh, conv = value_iteration(
        neighbors,
        costs,
        gamma=gamma,
    )

    return V, policy, Vh, Qh, conv


def run_policy_iteration(neighbors, costs, gamma):

    V, policy, Vh, Qh, conv = policy_iteration(
        neighbors,
        costs,
        gamma=gamma,
    )

    return V, policy, Vh, Qh, conv


def run_q_learning(neighbors, costs, start, goal, args):

    Q, policy, Vh, Qh, conv = q_learning(
        neighbors,
        costs,
        start,
        goal,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps=args.eps,
    )

    policy_nodes = []

    for s, a in enumerate(policy):

        if a is None:
            policy_nodes.append(None)
        else:
            policy_nodes.append(neighbors[s][a])

    V = np.max(Q, axis=1)

    return V, policy_nodes, Vh, Qh, conv


def run_q_learning_njit(neighbors, costs, start, goal, args):

    neighbors_arr, costs_arr, n_actions = pad_arrays(neighbors, costs)

    Q, policy, Vh, Qh, conv = q_learning_njit(
        neighbors_arr,
        costs_arr,
        n_actions,
        start,
        goal,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        eps=args.eps,
    )

    policy_nodes = []

    for s, a in enumerate(policy):

        if a < 0:
            policy_nodes.append(None)
        else:
            policy_nodes.append(neighbors[s][a])

    V = np.max(Q, axis=1)

    return V, policy_nodes, Vh, Qh, conv


def save_plot(fig, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig.savefig(path, bbox_inches="tight", dpi=200)


def main():

    args = parse_args()

    G, nodes, node_index, neighbors, costs, turns = build_graph(args)

    start = len(nodes) - 1
    goal = 0

    if args.method == "value":

        V, policy, Vh, Qh, conv = run_value_iteration(
            neighbors,
            costs,
            args.gamma,
        )

    elif args.method == "policy":

        V, policy, Vh, Qh, conv = run_policy_iteration(
            neighbors,
            costs,
            args.gamma,
        )

    elif args.method == "q":

        V, policy, Vh, Qh, conv = run_q_learning(
            neighbors,
            costs,
            start,
            goal,
            args,
        )

    else:

        V, policy, Vh, Qh, conv = run_q_learning_njit(
            neighbors,
            costs,
            start,
            goal,
            args,
        )

    fig1 = plot_convergence(Vh, Qh, conv)

    fig2 = visualize_graph(
        G,
        V,
        policy,
        nodes,
        node_index,
        turns,
        start=start,
        node_size=10 if args.graph == "city" else 250,
        show_node_id=(args.graph == "artificial"),
        arrowwidth=0.5 if args.graph == "city" else 2,
        show_trajectory_only=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    save_plot(fig1, f"{args.save_dir}/{args.method}_convergence.png")

    save_plot(fig2, f"{args.save_dir}/{args.method}_policy.png")


if __name__ == "__main__":
    # Example run: 
    # python run.py --method value
    # python run.py --method value --graph city
    # python run.py --method policy --graph city
    # python run.py --method q_njit --graph city

    main()