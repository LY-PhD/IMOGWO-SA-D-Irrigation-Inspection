"""
IMOGWO‑SA/D implementation for farmland irrigation canal inspection task allocation.
Simplified version for PhD‑thesis reproducibility only.
Random seeds are defined in "Random Seed List.csv".
This script runs for 10‑tasks / 50‑farmers test instance.
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def load_seed_by_instance(instance_name, run_id):
    """从 Random Seed List.csv 读取指定算例、指定轮次的种子"""
    with open("Random Seed List.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Test_instance"] == instance_name and int(row["Run_ID"]) == run_id:
                return int(row["Seed"])
    raise ValueError(f"Seed not found for {instance_name}, Run_ID={run_id}")

# ======================
# 选择要运行的算例与轮次
# ======================
TARGET_INSTANCE = "10/50"
RUN_ID = 1
seed_val = load_seed_by_instance(TARGET_INSTANCE, RUN_ID)
set_global_seed(seed_val)
print(f"Loaded seed: {seed_val}, Instance: {TARGET_INSTANCE}, Run: {RUN_ID}")

# ==============================================
# Global problem parameters: task, farmer, budget (10/50 instance)
# ==============================================
m = 10
n = 50
B = 500

tasks = []
for i in range(m):
    wt_s = random.randint(6, 12)
    task = {
        "task_id": i,
        "p_i": random.randint(50, 100),
        "e_i_req": random.choice([0.1, 0.2, 0.3]),
        "wt_s": wt_s,
        "wt_e": random.randint(wt_s + 4, 18),
        "t_ij_base": random.randint(1, 4)
    }
    tasks.append(task)

farmers = []
for j in range(n):
    e_j = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    r_j = random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    c_j_min = round(12 + 2 * e_j + 1.5 * r_j)
    c_j_prime = round(8 + 1.5 * e_j + 0.8 * r_j, 1)
    farmer = {
        "farmer_id": j,
        "e_j": e_j,
        "r_j": r_j,
        "c_j_min": c_j_min,
        "c_j": random.randint(1, 5),
        "c_j_prime": c_j_prime,
        "wf_s": random.randint(4, 6),
        "wf_e": random.randint(18, 20)
    }
    farmers.append(farmer)

d_ij = np.random.randint(1, 30, size=(m, n))

tau1, tau2 = 0.6, 0.4
q_ij = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        q_ij[i][j] = tau1 * farmers[j]["e_j"] + tau2 * farmers[j]["r_j"]

c_ij = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        c_ij[i][j] = farmers[j]["c_j_min"] + farmers[j]["c_j"] * d_ij[i][j]

t_ij = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        if farmers[j]["e_j"] >= tasks[i]["e_i_req"]:
            t_ij[i][j] = tasks[i]["t_ij_base"] * (tasks[i]["e_i_req"] / farmers[j]["e_j"])
        else:
            t_ij[i][j] = float('inf')

# ==============================================
# Algorithm hyper‑parameters
# ==============================================
N = 100
M = N
MaxIter = 50
T0 = 100
T_end = 1e-3
alpha_cool = 0.95
L = 10
omega_rand = 0.6

# ==============================================
# Constraint‑check functions
# ==============================================
def is_able(i, j):
    return farmers[j]["e_j"] >= tasks[i]["e_i_req"]

def time_ok(i, j, start_time):
    t_cost = t_ij[i][j]
    if start_time < farmers[j]["wf_s"] or start_time + t_cost > farmers[j]["wf_e"]:
        return False
    if start_time < tasks[i]["wt_s"] or start_time + t_cost > tasks[i]["wt_e"]:
        return False
    return True

def assign_unique(X):
    for i in range(m):
        if sum(X[i]) != 1:
            return False
    for j in range(n):
        if sum(X[:, j]) > 1:
            return False
    return True

def budget_ok(X):
    total = sum(X[i][j] * c_ij[i][j] for i in range(m) for j in range(n))
    return total <= B

def is_feasible(X, start_times):
    if not assign_unique(X):
        return False
    if not budget_ok(X):
        return False
    for i in range(m):
        j = np.argmax(X[i])
        if not is_able(i, j):
            return False
        if not time_ok(i, j, start_times[i][j]):
            return False
    return True

# ==============================================
# Objective functions
# ==============================================
def f1(X):
    total = 0
    for i in range(m):
        for j in range(n):
            if X[i][j] == 1:
                total += (tasks[i]["p_i"] - c_ij[i][j]) * q_ij[i][j]
    return -total

def f2(X):
    total = 0
    for i in range(m):
        for j in range(n):
            if X[i][j] == 1:
                total += farmers[j]["c_j_prime"] * t_ij[i][j] + farmers[j]["c_j"] * d_ij[i][j]
    return total

# ==============================================
# Solution initialization
# ==============================================
def init_solution():
    X = np.zeros((m, n), dtype=int)
    start_times = np.zeros((m, n))
    assigned_farmers = set()
    for i in range(m):
        feasible_js = []
        for j in range(n):
            if j not in assigned_farmers and is_able(i, j):
                s = max(farmers[j]["wf_s"], tasks[i]["wt_s"])
                if s + t_ij[i][j] <= min(farmers[j]["wf_e"], tasks[i]["wt_e"]):
                    feasible_js.append((j, s))
        if not feasible_js:
            return init_solution()
        j, s = random.choice(feasible_js)
        X[i][j] = 1
        start_times[i][j] = s
        assigned_farmers.add(j)
    if not is_feasible(X, start_times):
        return init_solution()
    return X, start_times

# ==============================================
# Population initialization
# ==============================================
Pop = []
StartTimes = []
key_set = set()
while len(Pop) < N:
    x, st = init_solution()
    key = x.tobytes()
    if key not in key_set:
        key_set.add(key)
        Pop.append(x)
        StartTimes.append(st)

# ==============================================
# Weight vectors & neighborhood construction
# ==============================================
weights = np.linspace(0, 1, M)
lambda_vectors = np.vstack([weights, 1 - weights]).T
neighborhood = []
for k in range(M):
    dist = np.sum((lambda_vectors - lambda_vectors[k]) ** 2, axis=1)
    neighbors = np.argsort(dist)[:L]
    neighborhood.append(neighbors.tolist())

# ==============================================
# Ideal point initialization
# ==============================================
z1 = min(f1(x) for x in Pop)
z2 = min(f2(x) for x in Pop)
z_star = [z1, z2]

def g(k, X):
    l1, l2 = lambda_vectors[k]
    return max(l1 * abs(f1(X) - z_star[0]), l2 * abs(f2(X) - z_star[1]))

# ==============================================
# Dominance, crowding‑distance, archive maintenance
# ==============================================
def is_dominates(X, Y):
    f1x, f2x = f1(X), f2(X)
    f1y, f2y = f1(Y), f2(Y)
    return (f1x <= f1y and f2x <= f2y) and (f1x < f1y or f2x < f2y)

def crowding_distance(solutions):
    n = len(solutions)
    if n == 0:
        return []
    f1_list = [f1(x) for x in solutions]
    f2_list = [f2(x) for x in solutions]
    f1_min, f1_max = min(f1_list), max(f1_list)
    f2_min, f2_max = min(f2_list), max(f2_list)
    f1_range = f1_max - f1_min if f1_max != f1_min else 1
    f2_range = f2_max - f2_min if f2_max != f2_min else 1
    sorted_idx = np.argsort(f1_list)
    dist = np.zeros(n)
    dist[sorted_idx[0]] = np.inf
    dist[sorted_idx[-1]] = np.inf
    for i in range(1, n - 1):
        dist[sorted_idx[i]] = (abs(f1_list[sorted_idx[i + 1]] - f1_list[sorted_idx[i - 1]]) / f1_range +
                               abs(f2_list[sorted_idx[i + 1]] - f2_list[sorted_idx[i - 1]]) / f2_range)
    return dist

def update_archive(archive, pop):
    combined = archive + pop
    non_dominated = []
    for x in combined:
        dominated = False
        for y in combined:
            if is_dominates(y, x):
                dominated = True
                break
        if not dominated:
            non_dominated.append(x)
    unique = []
    for s in non_dominated:
        if not any(np.array_equal(s, u) for u in unique):
            unique.append(s)
    if len(unique) > N:
        dist = crowding_distance(unique)
        sorted_idx = np.argsort(dist)[::-1]
        unique = [unique[i] for i in sorted_idx[:N]]
    return unique

A = update_archive([], Pop)

# ==============================================
# Main iteration loop
# ==============================================
iter = 0
T = T0
print("=====================================")
print("  IMOGWO‑SA/D running ")
print("=====================================")
while iter < MaxIter and T > T_end:
    if iter % 10 == 0 and iter > 0:
        perturb_num = int(M * 0.1)
        perturb_idx = random.sample(range(M), perturb_num)
        for idx in perturb_idx:
            x, st = init_solution()
            Pop[idx] = x
            StartTimes[idx] = st

    for k in range(M):
        S_N = [Pop[idx] for idx in neighborhood[k]]
        scores = [g(k, x) for x in S_N]
        sorted_idx = np.argsort(scores)
        alpha = S_N[sorted_idx[0]]
        beta = S_N[sorted_idx[1]]
        delta = S_N[sorted_idx[2]]

        X_new = Pop[k].copy()
        st_new = StartTimes[k].copy()

        w_alpha = max(0.1, 1.5 - iter / MaxIter)
        w_beta = max(0.1, w_alpha - 0.4)
        w_delta = max(0.1, w_beta - 0.4)

        assigned_j = set(np.argmax(X_new, axis=1))
        for i in range(m):
            ja = np.argmax(alpha[i])
            jb = np.argmax(beta[i])
            jd = np.argmax(delta[i])

            F_i = []
            for j in range(n):
                if j not in assigned_j and is_able(i, j):
                    s = max(farmers[j]["wf_s"], tasks[i]["wt_s"])
                    if s + t_ij[i][j] <= min(farmers[j]["wf_e"], tasks[i]["wt_e"]):
                        F_i.append((j, s))
            if not F_i:
                continue

            prob = []
            for j, s in F_i:
                pa = w_alpha * (1 if j == ja else 0)
                pb = w_beta * (1 if j == jb else 0)
                pd = w_delta * (1 if j == jd else 0)
                pr = omega_rand / len(F_i)
                prob.append(max(0.0, pa + pb + pd + pr))
            prob = np.array(prob)
            s_prob = prob.sum()
            if s_prob < 1e-8:
                prob = np.ones_like(prob) / len(prob)
            else:
                prob /= s_prob

            chosen = np.random.choice(len(F_i), p=prob)
            j_sel, s_sel = F_i[chosen]

            old_j = np.argmax(X_new[i])
            X_new[i] = 0
            X_new[i][j_sel] = 1
            st_new[i][j_sel] = s_sel
            assigned_j.remove(old_j)
            assigned_j.add(j_sel)

        # SA perturbation
        X_pert = X_new.copy()
        st_pert = st_new.copy()
        if m >= 2:
            i1, i2 = random.sample(range(m), 2)
            j1 = np.argmax(X_pert[i1])
            j2 = np.argmax(X_pert[i2])
            s1 = max(farmers[j2]["wf_s"], tasks[i1]["wt_s"])
            s2 = max(farmers[j1]["wf_s"], tasks[i2]["wt_s"])
            if (s1 + t_ij[i1][j2] <= min(farmers[j2]["wf_e"], tasks[i1]["wt_e"]) and
                    s2 + t_ij[i2][j1] <= min(farmers[j1]["wf_e"], tasks[i2]["wt_e"])):
                X_pert[i1] = 0
                X_pert[i1][j2] = 1
                X_pert[i2] = 0
                X_pert[i2][j1] = 1
                st_pert[i1][j2] = s1
                st_pert[i2][j1] = s2

        dg = g(k, X_pert) - g(k, X_new)
        if dg < 0 or random.random() < np.exp(-dg / T):
            X_new = X_pert
            st_new = st_pert

        if not is_feasible(X_new, st_new):
            X_new, st_new = init_solution()

        Pop[k] = X_new
        StartTimes[k] = st_new

    A = update_archive(A, Pop)
    z_star[0] = min(f1(x) for x in A)
    z_star[1] = min(f2(x) for x in A)
    T *= alpha_cool
    iter += 1
    print(f"Iter {iter:3d} | Pareto count: {len(A):2d} | Temperature: {T:6.2f}")

# ==============================================
# Output and visualization
# ==============================================
print("\n" + "=" * 60)
print("                Final Pareto Solutions               ")
print("=" * 60)
cost_list = []
profit_list = []
for idx, sol in enumerate(A):
    cost_val = f2(sol)
    profit_val = -f1(sol)
    cost_list.append(cost_val)
    profit_list.append(profit_val)
    print(f"\n--- Solution {idx + 1} ---")
    print(f"Platform utility: {profit_val:.2f} | Farmer‑side cost: {cost_val:.2f}")
    for i in range(m):
        j = np.argmax(sol[i])
        print(f"Task{i} → Farmer{j}")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

sorted_pairs = sorted(zip(profit_list, cost_list))
sorted_profit, sorted_cost = zip(*sorted_pairs)
plt.figure(figsize=(10, 6))
plt.scatter(sorted_profit, sorted_cost, c='red', s=80, label='Pareto non‑dominated solutions')
plt.plot(sorted_profit, sorted_cost, 'r--', linewidth=2, label='Pareto Front')
plt.xlabel('F1', fontsize=12)
plt.ylabel('F2', fontsize=12)
plt.title('IMOGWO‑SA/D Pareto Front', fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig('Pareto_front(1).png', dpi=300, bbox_inches='tight')
plt.show()
