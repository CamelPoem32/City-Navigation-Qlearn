import numpy as  np

def value_iteration(neighbors, costs, gamma=0.95, cost_per_step=40.0, track_history=True, iter_after_conv=1):

    N = len(neighbors)

    V = np.ones(N) * (cost_per_step / (1 - gamma))  # Upper bound
    V[0] = 0
    conv_iter = 0

    V_history = []
    Q_history = []
    conv = []

    while True:

        delta = 0
        Q = []
        V_old = V.copy()

        for s in range(N):

            if len(neighbors[s]) == 0:
                Q.append([])
                continue

            # Bellman update with costs
            q = -costs[s] + gamma * V_old[neighbors[s]]

            v_new = np.max(q)

            if s == 0: 
                v_new = 0

            delta = max(delta, abs(V_old[s] - v_new))

            V[s] = v_new

            Q.append(q)

        if track_history:
            V_history.append(V.copy())
            Q_history.append(Q.copy())
            conv.append(delta)

        if delta < 1e-6 and iter_after_conv >= conv_iter:
            break
        elif delta < 1e-6: conv_iter += 1
        else: iter_after_conv = 0

    policy = []

    for s in range(N):

        if len(neighbors[s]) == 0:
            policy.append(None)
            continue

        q = -costs[s] + gamma * V[neighbors[s]]

        policy.append(neighbors[s][np.argmax(q)])

    policy[0] = None

    return V, policy, V_history, Q_history, conv

def policy_evaluation(policy, neighbors, costs, gamma=0.99, track_history=False, cost_per_step=40.0):

    N = len(neighbors)

    V = np.ones(N) * (cost_per_step / (1 - gamma))
    # V = np.zeros(N)

    V[0] = 0

    V_history = []

    for _ in range(1000):

        for s in range(1, N):

            a = policy[s]

            if a is None:
                continue

            idx = np.where(neighbors[s] == a)[0][0]

            V[s] = -costs[s][idx] + gamma * V[a]

        if track_history:
            V_history.append(V.copy())

    V[0] = 0

    if track_history:
        return V, V_history
    else:
        return V    
    
def policy_iteration(neighbors, costs, gamma=0.99, track_history=True):

    N = len(neighbors)

    policy = []

    for s in range(N):
        if len(neighbors[s]) == 0:
            policy.append(None)
        else:
            policy.append(neighbors[s][0])

    V_history = []
    Q_history = []
    conv = []

    while True:

        V = policy_evaluation(policy, neighbors, costs, gamma)

        stable = True
        delta = 0

        Q = []

        for s in range(N):

            if len(neighbors[s]) == 0:
                Q.append([])
                continue

            q = -costs[s] + gamma * V[neighbors[s]]

            best_next = neighbors[s][np.argmax(q)]

            if best_next != policy[s]:
                stable = False

            delta = max(delta, np.max(np.abs(q)))

            policy[s] = best_next

            Q.append(q)

        if track_history:
            V_history.append(V.copy())
            Q_history.append(Q.copy())
            conv.append(delta)

        if stable:
            break

    policy[0] = None

    return V, policy, V_history, Q_history, conv

def q_learning(neighbors, costs, start, goal,
               episodes=2000,
               alpha=0.1,
               gamma=0.99,
               eps=0.1,
               track_history=True, 
               cost_per_step=40.0):

    N = len(neighbors)
    max_actions = max((len(n) for n in neighbors if any(n)), default=0)

    # Q = np.ones((N, max_actions)) * (cost_per_step / (1 - gamma))
    # Q[0, :] = 0
    Q = np.zeros((N, max_actions))

    V_history = []
    Q_history = []
    conv = []

    for ep in range(episodes):

        Q_old = Q.copy()

        s = start

        while s != goal:

            # epsilon-greedy exploration
            if np.random.rand() < eps:
                a_idx = np.random.randint(len(neighbors[s]))
            else:
                a_idx = np.argmax(Q[s, :len(neighbors[s])])

            s_next = neighbors[s][a_idx]

            r = -costs[s][a_idx]

            # future value
            if s_next == goal:
                max_future_q = 0.0
            else:
                max_future_q = np.max(Q[s_next, :len(neighbors[s_next])])

            # Q-learning update
            Q[s, a_idx] += alpha * (r + gamma * max_future_q - Q[s, a_idx])

            s = s_next

        # ---- record history ----
        if track_history:

            V = np.zeros(N)

            for i in range(N):

                na = len(neighbors[i])

                if na == 0:
                    continue

                V[i] = np.max(Q[i, :na])

            V_history.append(V.copy())
            Q_history.append(Q.copy())

            conv.append(np.max(np.abs(Q - Q_old)))

    # ---- policy extraction ----
    policy = []

    for s in range(N):

        n_actions = len(neighbors[s])

        if n_actions == 0 or s == goal:
            policy.append(None)
        else:
            policy.append(np.argmax(Q[s, :n_actions]))

    policy = np.array(policy, dtype=object)

    policy[0] = None

    return Q, policy, V_history, Q_history, conv

from numba import njit, prange


def pad_arrays(neighbors, costs):

    N = len(neighbors)
    max_actions = max(len(n) for n in neighbors)

    neigh = -np.ones((N,max_actions), dtype=np.int64)
    c = np.zeros((N,max_actions))
    n_actions = np.zeros(N, dtype=np.int64)

    for i in range(N):

        na = len(neighbors[i])
        n_actions[i] = na

        neigh[i,:na] = neighbors[i]
        c[i,:na] = costs[i]

    return neigh, c, n_actions

@njit
def q_learning_njit(neighbors,
               costs,
               n_actions,
               start,
               goal,
               episodes,
               alpha,
               gamma,
               eps,
               max_iters=1_000_000,
               cost_per_step = 40.0):

    N = neighbors.shape[0]
    max_actions = neighbors.shape[1]

    # Q = np.ones((N, max_actions)) * (cost_per_step / (1 - gamma))
    # Q[0, :] = 0
    Q = np.zeros((N, max_actions))

    V_history = np.zeros((episodes, N))
    Q_history = np.zeros((episodes, N, max_actions))
    conv = np.zeros(episodes)

    for ep in range(episodes):

        Q_old = Q.copy()

        s = start
        iter = 0

        while s != goal:

            iter += 1
            if iter >= max_iters: break

            na = n_actions[s]

            if na == 0:
                break

            if np.random.rand() < eps:
                a = np.random.randint(na)
            else:
                best = 0
                best_val = Q[s,0]
                for i in range(1,na):
                    if Q[s,i] > best_val:
                        best_val = Q[s,i]
                        best = i
                a = best

            s_next = neighbors[s,a]

            r = -costs[s,a]

            if s_next == goal:
                max_future_q = 0.0
            else:
                na_next = n_actions[s_next]

                max_future_q = Q[s_next,0]

                for i in range(1,na_next):
                    if Q[s_next,i] > max_future_q:
                        max_future_q = Q[s_next,i]

            Q[s,a] += alpha * (r + gamma * max_future_q - Q[s,a])

            s = s_next

        # record history
        for i in range(N):
            best = Q[i,0]-1000
            for j in range(0,max_actions):
                if Q[i,j] > best:
                    best = Q[i,j]
            V_history[ep,i] = best

        Q_history[ep,:,:] = Q

        maxdiff = 0.0
        for i in range(N):
            for j in range(max_actions):
                d = abs(Q[i,j]-Q_old[i,j])
                if d > maxdiff:
                    maxdiff = d

        conv[ep] = maxdiff

    policy = np.empty(N, dtype=np.int64)

    for s in range(N):

        na = n_actions[s]

        if na == 0 or s == goal:
            policy[s] = -1
        else:

            best = 0
            best_val = Q[s,0]

            for i in range(1,na):
                if Q[s,i] > best_val:
                    best_val = Q[s,i]
                    best = i

            policy[s] = best

    policy[0] = np.nan

    return Q, policy, V_history, Q_history, conv