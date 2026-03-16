# Risk-Aware Logistics Routing with Reinforcement Learning

## Overview

This project studies **route planning for logistics vehicles** under safety constraints using **Dynamic Programming and Reinforcement Learning**.

The central idea is that **left turns are more dangerous and slower than right turns**, therefore an optimal routing policy should prefer routes that:

- minimize travel time
- minimize crash probability
- reduce risky maneuvers (especially left turns)

The problem is formulated as a **Markov Decision Process (MDP)** on a **road graph** and solved using three algorithms:

- Value Iteration
- Policy Iteration
- Q-Learning

The road network is either:

- artificially generated for experiments, or
- downloaded from **OpenStreetMap** using `osmnx`.

---

# Problem Formulation

We model the routing task as a **finite Markov Decision Process**

$$
\mathcal{M} = (S, A, P, R, \gamma)
$$

where

- $S$ is the set of states
- $A$ is the set of actions
- $P(s'|s,a)$ is the transition probability
- $R(s,a)$ is the reward function
- $\gamma \in (0,1)$ is the discount factor

The objective is to find an **optimal policy**

$$
\pi{}^*(s) = \arg\max_a Q*(s,a)
$$

that maximizes the expected cumulative discounted reward

$$
\mathbb{E}\left[\sum_{t=0}^{\infty}\gamma^t r_t\right].
$$

---

# Environment Representation

The road network is represented as a **directed graph**

$$
G = (V, E)
$$

where

- $V$ is the set of nodes (intersections)
- $E$ is the set of directed edges (road segments)

Each edge represents a **legal driving direction between intersections**.

---

# State Space

The **state space** consists of all intersections in the road graph.

$$
S = \{s_1, s_2, ..., s_N\}
$$

where $s_i \in V$ represents the **vehicle located at intersection $i$**.

Thus the total number of states is $|S| = |V|.$

Each state contains:

- geographic coordinates
- outgoing road connections
- cost of each outgoing action

---

# Action Space

At each state the agent may choose any **outgoing road segment**.

The action set at state $s$ is

$$
A(s) = \{a_1, a_2, ..., a_{k_s}\}
$$

where $k_s$ is the number of outgoing roads from intersection $s$.

Each action corresponds to selecting a **neighbor node**

$$
a = (s \rightarrow s')
$$

so that

$$
s' \in \text{Neighbors}(s).
$$

Formally,

$$
A(s) = \{ s' \mid (s,s') \in E \}.
$$

Thus the agent's decision at each step is

$$
a_t \in A(s_t).
$$

---

# Transition Model

Transitions are **deterministic**.

If the agent selects action $$a = (s \rightarrow s')$$ then

$$
P(s'|s,a) = 1.
$$

Thus

$$
s_{t+1} = s'.
$$

---

# Turn Classification

The system classifies maneuvers as

- left turn
- right turn
- straight movement

using the **orientation of three consecutive nodes**

$$
(s_{t-1}, s_t, s_{t+1}).
$$

Let

$$
v_1 = s_t - s_{t-1}
$$

$$
v_2 = s_{t+1} - s_t
$$

The cross product

$$
z = v_1 \times v_2
$$

determines the turn type:

- $z > 0$ → left turn
- $z < 0$ → right turn
- $|z| \approx 0$ → straight

---

# Cost Model

Each action has an associated **cost**

$$
c(s,a).
$$

The cost reflects:

- travel time
- crash probability

Default parameters are

| Turn Type | Time | Crash Probability |
|----------|------|------------------|
| Right | 20 | 0.001 |
| Straight | 40 | 0.003 |
| Left | 60 | 0.01 |

The total cost is

$$
c(s,a) = t(s,a) + \lambda p(s,a)
$$

where

- $t(s,a)$ is travel time
- $p(s,a)$ is crash probability
- $\lambda$ is a weighting coefficient.

---

# Reward Function

Reinforcement learning algorithms maximize reward.

Since the objective is **minimizing cost**, reward is defined as

$$
R(s,a) = -c(s,a).
$$

Thus lower-cost actions yield higher rewards.

---

# Goal State

The goal state $g$ represents the destination.

When the agent reaches $g$ the episode terminates and

$$
V(g) = 0.
$$

---

# Solution Methods

## Value Iteration

Value iteration computes the optimal value function by repeatedly applying the **Bellman optimality update**

$$
V_{k+1}(s) =
\max_{a \in A(s)}
\left[
R(s,a) + \gamma V_k(s')
\right].
$$

Substituting the reward definition

$$
V_{k+1}(s) =
\max_a
\left[
-c(s,a) + \gamma V_k(s')
\right].
$$

The optimal policy is extracted as

$$
\pi(s) =
\arg\max_a
\left[
-c(s,a) + \gamma V(s')
\right].
$$

---

## Policy Iteration

Policy iteration alternates between two steps.

### Policy Evaluation

For a fixed policy $\pi$:

$$
V^{\pi}(s) =
R(s,\pi(s)) + \gamma V^{\pi}(s').
$$

### Policy Improvement

The policy is updated by

$$
\pi_{new}(s) =
\arg\max_a
\left[
R(s,a) + \gamma V^{\pi}(s')
\right].
$$

The process repeats until the policy stabilizes.

---

## Q-Learning

Q-learning is a **model-free reinforcement learning algorithm** that learns action values directly from experience.

The update rule is

$$
Q(s,a) \leftarrow Q(s,a)+\alpha \left[r+\gamma \max_{a'}Q(s',a') -  (s,a) \right].
$$

where

- $\alpha$ is the learning rate
- $\gamma$ is the discount factor.

The policy is

$$
\pi(s) = \arg\max_a Q(s,a).
$$

Exploration is implemented using **ε-greedy action selection**

$$
a =
\begin{cases}
\text{random action} & \text{with probability } \varepsilon \\
\arg\max_a Q(s,a) & \text{otherwise}
\end{cases}
$$

---

# Visualization

The project includes visualization tools for

- road graphs
- value functions
- optimal policies
- turn classifications

Nodes are colored according to the value function

$$
V(s).
$$

Edges representing the policy are drawn as **red arrows**.

Turn types can also be shown as symbols:

- left turn ←
- right turn →
- straight ↑

---

# Experimental Workflow

1. Generate or download graph
`G = create_artificial_graph(5)`
or 
`G = create_city_graph("Piedmont, California, USA")`

2. Detect turn directions
`turns = detect_turns(G)`

3. Convert graph to NumPy representation
`nodes, node_index, coords, neighbors = graph_to_numpy(G)`

4. Compute action costs
`neighbors, costs = build_neighbors_costs(G, nodes, node_index, turns)`

5. Run algorithms
`V, policy = value_iteration(neighbors, costs)`
or
`V, policy = policy_iteration(neighbors, costs)`
or
`Q, policy = q_learning(neighbors, costs)`

6. Visualize policy
`visualize_graph(...)`

---

# Results

Experiments show the following behavior.

Policy Iteration converges in very few iterations because each step directly improves the policy.

Value Iteration converges more slowly because it repeatedly updates state values before extracting the policy.

Q-Learning requires many episodes but has the advantage of **not requiring a known model of the environment**.

All methods successfully produce routing strategies that prefer

- right turns
- shorter travel time
- lower crash risk

which aligns with real logistics routing strategies.

---

# Conclusion

This project demonstrates how **reinforcement learning and dynamic programming methods** can be applied to routing problems on real road networks.

The key contributions are:

- modeling intersections as MDP states
- incorporating safety into routing costs
- solving the resulting problem using RL algorithms
- visualizing optimal policies on real city graphs

The framework is flexible and can be extended to more realistic logistics problems.

---

# Dependencies

```
numpy
networkx
matplotlib
osmnx
numba
```

Install with

```
pip install numpy networkx matplotlib osmnx numba
```

## Running Experiments

Experiments are executed through the `run.py` script, which provides a simple CLI interface for selecting the environment and the reinforcement learning algorithm.

### General Usage

```bash
python run.py --method <algorithm> --graph <type> [additional options]
```

### Available Algorithms

| Method | Description |
|------|------|
| `value` | Value Iteration |
| `policy` | Policy Iteration |
| `q` | Q-learning |
| `q_njit` | Q-learning accelerated with Numba |

### Graph Types

| Graph | Description |
|------|------|
| `artificial` | Synthetic grid-like road network |
| `city` | Real road network downloaded from OpenStreetMap |

### Example Runs

Run **Value Iteration** on a small artificial road network:

```bash
python run.py --method value --graph artificial --size 5
```

Run **Policy Iteration** on a real city map:

```bash
python run.py --method policy --graph city --place "Piedmont, California, USA"
```

Run **Q-learning** with trajectory sampling:

```bash
python run.py --method q --graph city --episodes 5000
```

Run **Numba-accelerated Q-learning**:

```bash
python run.py --method q_njit --graph city --episodes 5000
```

### Generated Outputs

Each run produces two figures:

- **Policy visualization** on the road graph
- **Convergence plots** showing value and Q-function evolution

Results are saved to the `results/` directory:

```
results/
    value_convergence.png
    value_policy.png
    q_convergence.png
    q_policy.png
```

---

## Collision Probability Experiment

To model traffic safety more realistically, we extended the environment with **collision probabilities** for each state-action pair.

Let  

$$p_{\text{crash}}(s,a)$$  

denote the probability that executing action $a$ in state $s$ results in a collision.  
If a crash occurs, the episode terminates and a large penalty is assigned:

$$
r =
\begin{cases}
-10^6, & \text{if crash occurs} \\
-c(s,a), & \text{otherwise}
\end{cases}
$$

The transition therefore becomes stochastic:

$$
s_{t+1} =
\begin{cases}
\text{terminal crash state}, & \text{with probability } p_{\text{crash}}(s,a) \\
s', & \text{with probability } 1 - p_{\text{crash}}(s,a)
\end{cases}
$$

### Observed Limitation

During testing we attempted to enforce extremely unsafe maneuvers by setting

$$
p_{\text{crash}}(\text{left turn}) = 1
$$

which theoretically should make any trajectory containing a left turn impossible.  
However, learned policies occasionally still contained paths that effectively produced a left turn.

This behavior likely arises from the **state representation used in the Q-table**.

In the current formulation:

- state $s$ corresponds only to a **node (intersection)** in the road graph
- actions correspond to **outgoing edges**

However, determining whether a maneuver is **left, right, or straight** depends on the **previous edge taken**.  
Thus the turn type actually depends on a **three-node configuration**:

$$
(v_{t-1}, v_t, v_{t+1})
$$

while the Q-table only stores:

$$
Q(v_t, a)
$$

Because the previous node $$v_{t-1}$$ is not encoded in the state, the agent cannot fully distinguish all turn configurations. As a result, some sequences of actions may still form an effective left turn even when individual edge transitions appear safe.

A more precise formulation would require **augmenting the state space** to include the previous node:

$$
s_t = (v_{t-1}, v_t)
$$

which would allow the agent to explicitly represent turn geometry and correctly penalize unsafe maneuvers.

Despite this limitation, the current approach still demonstrates how reinforcement learning can be applied to **risk-aware routing in urban road networks**.