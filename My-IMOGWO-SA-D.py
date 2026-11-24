# ==============================================
# random：用于随机选择、随机初始化
# numpy：用于矩阵运算、数值计算
# matplotlib：用于画 Pareto 图
# ==============================================
import random
import numpy as np
import matplotlib.pyplot as plt

# ==============================================
# 1. 【全局参数设置】：任务数 m、农户数 n、总预算 B
# ==============================================
m = 100  # 任务总数
n = 400  # 农户总数
B = 5000  # 平台总预算（约束C5）

# ==============================================
# 任务参数定义（完全匹配论文符号）
# p_i：任务i给平台带来的收益
# e_i_req：任务i要求的最低设备能力
# wt_s, wt_e：任务i的时间窗（开始~结束）
# t_ij_base：完成任务i需要的基础时间
# ==============================================
# 生成一个包含 m 个任务的列表，每个任务都是一个字典
tasks = []
for i in range(m):
    wt_s = random.randint(6, 14)  # 先生成变量
    task = {
        "task_id": i,
        "p_i": random.randint(50, 150),    # 任务收益(元）
        "e_i_req": random.randint(2, 6),   # 任务能力要求
        "wt_s": wt_s,                            # 任务可开始时间
        "wt_e": random.randint(wt_s + 4, 22),  # 任务必须完成时间
        "t_ij_base": random.randint(1, 4)   # 任务基础耗时
    }
    tasks.append(task)

# ==============================================
# 农户参数定义（完全匹配论文符号）
# e_j：农户j的设备能力
# r_j：农户j的信誉分
# c_j_min：基础报酬
# c_j：单位距离成本
# c_j_prime：单位时间成本
# wf_s/wf_e：农户j的可工作时间窗
# ==============================================
# 生成一个包含 n 个农户的列表，每个农户都是一个字典
farmers = []
for j in range(n):
    e_j = random.randint(1, 10)       # 农户能力
    r_j = round(random.uniform(0.1, 0.9), 1)  # 农户信誉

    # 1. 基础报酬（和能力、信誉正相关，差距温和）
    c_j_min = round(10 + 2 * e_j + 1 * r_j)  # 四舍五入保留指定位数小数的函数，n=0即四舍五入后取整数

    # 2. 单位时间成本 也联动！！！
    c_j_prime = round(5 + 1.5 * e_j + 0.8 * r_j, 1)  # n=1即小数位保留1位

    farmer = {
        "farmer_id": j,
        "e_j": e_j,
        "r_j": r_j,
        "c_j_min": c_j_min,    # 基础报酬
        "c_j": random.randint(1, 3),  # 单位距离成本
        "c_j_prime": c_j_prime,            # 单位时间成本（已联动）
        "wf_s": random.randint(5, 8),
        "wf_e": random.randint(18, 24)
    }
    farmers.append(farmer)

# ==============================================
# 距离矩阵 d_ij：任务i到农户j的距离
# ==============================================
d_ij = np.random.randint(1, 11, size=(m, n))

# ==============================================
# 预期完成质量系数 q_ij（论文公式）
# q_ij = τ1*e_j + τ2*r_j
# ==============================================
tau1, tau2 = 0.6, 0.3
q_ij = np.zeros((m, n)) #创建一个 m 行 n 列的全 0 矩阵，用来存放每个任务对应所有农户的任务完成率。
for i in range(m):
    for j in range(n):
        q_ij[i][j] = tau1 * farmers[j]["e_j"] + tau2 * farmers[j]["r_j"]

# ==============================================
# 任务报酬 c_ij（论文公式）
# c_ij = c_j_min + c_j * d_ij
# ==============================================
c_ij = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        c_ij[i][j] = farmers[j]["c_j_min"] + farmers[j]["c_j"] * d_ij[i][j]

# ==============================================
# 任务执行时间 t_ij（论文约束：能力越高耗时越短）
# ==============================================
t_ij = np.zeros((m, n))
for i in range(m):
    for j in range(n):
        if farmers[j]["e_j"] >= tasks[i]["e_i_req"]:
            t_ij[i][j] = tasks[i]["t_ij_base"] * (tasks[i]["e_i_req"] / farmers[j]["e_j"])
        else:
            t_ij[i][j] = float('inf')  # 能力不够，时间无穷大

# ==============================================
# 算法参数（IMOGWO-SA/D）
# ==============================================
N = 100  # 种群规模
M = N    # 子问题数量（ = 种群规模）
MaxIter = 300  # 最大迭代次数
T0 = 100  # 模拟退火初始温度
T_end = 1e-3  # 退火终止温度 = 0.001
alpha_cool = 0.95  # 冷却系数
L = 15  # 邻域大小
omega_rand = 0.55  # 随机探索系数；0.4~0.5：平衡搜索与收敛，最常用、最推荐

# ==============================================
# 2. 【约束检查函数】—— 完全对应论文5大约束
# 这是算法最核心的“合法性判断”
# ==============================================

# 约束C3：能力匹配
def is_able(i, j):
    return farmers[j]["e_j"] >= tasks[i]["e_i_req"]

# 约束C4：时间窗匹配（农户 + 任务）
def time_ok(i, j, start_time):
    t_cost = t_ij[i][j]
    if start_time < farmers[j]["wf_s"] or start_time + t_cost > farmers[j]["wf_e"]:
        return False
    if start_time < tasks[i]["wt_s"] or start_time + t_cost > tasks[i]["wt_e"]:
        return False
    return True

# 约束C1+C2：任务唯一分配、农户不超载
def assign_unique(X):
    for i in range(m):
        if sum(X[i]) != 1:
            return False
    for j in range(n):
        if sum(X[:, j]) > 1:
            return False
    return True

# 约束C5：总预算不超限
def budget_ok(X):
    total = sum(X[i][j] * c_ij[i][j] for i in range(m) for j in range(n))
    return total <= B

# 综合约束判断：解是否合法
def is_feasible(X, start_times):
    if not assign_unique(X):  # 分配唯一性约束检查
        return False
    if not budget_ok(X):      # 预算约束检查
        return False
    for i in range(m):
        j = np.argmax(X[i])    # 返回数组中最大值所在的位置（索引 / 下标），不返回值本身
        if not is_able(i, j):  # 能力约束检查
            return False
        if not time_ok(i, j, start_times[i][j]):   # 时间窗约束检查
            return False
    return True

# ==============================================
# 3. 【目标函数】—— 严格对应论文
# ==============================================

# 目标1：最大化平台净收益 → 转为最小化 f1 = -收益
def f1(X):
    total = 0
    for i in range(m):
        for j in range(n):
            if X[i][j] == 1:
                total += (tasks[i]["p_i"] - c_ij[i][j]) * q_ij[i][j]
    return -total

# 目标2：最小化农户总成本（时间+距离）
def f2(X):
    total = 0
    for i in range(m):
        for j in range(n):
            if X[i][j] == 1:
                total += farmers[j]["c_j_prime"] * t_ij[i][j] + farmers[j]["c_j"] * d_ij[i][j]
    return total

# ==============================================
# 4. 【初始化可行解】
# 每执行一次，生成一个满足所有约束的任务分配方案，返回一组 (X, start_times)
# ==============================================
def init_solution():
    X = np.zeros((m, n), dtype=int)  # 新建空的分配矩阵 X
    start_times = np.zeros((m, n))  # 新建空的时间矩阵 start_times，存放每个任务的开始时间
    assigned_farmers = set()  # 创建一个空集合，用来记录已经分配过任务的农民编号，确保一个农民只分配一次

    for i in range(m):
        feasible_js = [] # 创建一个空列表，用来存放当前任务可以分配的、所有合法的农民编号。
        for j in range(n):
            if j not in assigned_farmers and is_able(i, j):
                s = max(farmers[j]["wf_s"], tasks[i]["wt_s"])
                if s + t_ij[i][j] <= min(farmers[j]["wf_e"], tasks[i]["wt_e"]):
                    feasible_js.append((j, s)) # 把 (农民编号，开始时间) 这个元组添加进列表
                    # feasible_js 里就是当前任务所有可选的合法农户（可能有多个）
        if not feasible_js: # 空列表 [] 被当作 False
            return init_solution()  # 无可行农户 → 重新生成 递归调用

        j, s = random.choice(feasible_js)
        X[i][j] = 1
        start_times[i][j] = s
        assigned_farmers.add(j)

    if not is_feasible(X, start_times):  # 综合约束判断：解是否合法
        return init_solution() #在函数内重新调用 init_solution()：任务、农民数据不变，只重新随机分配。
    return X, start_times

# ==============================================
# 5. 【初始化种群】
# ==============================================
Pop = []  # 新建空列表，存 N 个不重复的分配矩阵X
StartTimes = []  # 新建空列表，对应 N 个解的开始时间矩阵start_times
key_set = set()  # 新建空集合，用来判断解是否重复

# 一直生成，直到凑够 N个不重复解
while len(Pop) < N:
    x, st = init_solution()  # 生成一个解
    key = x.tobytes()  # 生成唯一标识

    if key not in key_set:  # 如果不重复
        key_set.add(key)
        Pop.append(x)
        StartTimes.append(st)

# ==============================================
# 6. 【生成均匀权重向量 + 邻域构建】（Decomposition 核心）
# ==============================================
weights = np.linspace(0, 1, M)  # 0 到 1 均匀分 M 份
lambda_vectors = np.vstack([weights, 1 - weights]).T

neighborhood = []  # 新建空列表，用来保存所有子问题的邻居编号
for k in range(M):
    dist = np.sum((lambda_vectors - lambda_vectors[k]) ** 2, axis=1)  # 第 k 个权重向量和其他所有向量的欧氏距离平方
    neighbors = np.argsort(dist)[:L]  # 取到当前子问题的邻域集合
    neighborhood.append(neighbors.tolist())

# ==============================================
# 7. 【初始化理想点 z*】
# ==============================================
z1 = min(f1(x) for x in Pop)
z2 = min(f2(x) for x in Pop)
z_star = [z1, z2]

# ==============================================
# 8. 【切比雪夫分解函数】
# g^k(X) = max{ λ1|f1-z1*|, λ2|f2-z2*| }
# ==============================================
def g(k, X):
    l1, l2 = lambda_vectors[k]
    return max(l1 * abs(f1(X) - z_star[0]), l2 * abs(f2(X) - z_star[1]))

# ==============================================
# 9. 【非支配判断 + 拥挤距离 + 存档更新】
# ==============================================

# 判断解X是否支配解Y
def is_dominates(X, Y):
    f1x, f2x = f1(X), f2(X)
    f1y, f2y = f1(Y), f2(Y)
    return (f1x <= f1y and f2x <= f2y) and (f1x < f1y or f2x < f2y)

# 拥挤距离计算（保持Pareto解分布均匀）： 拥挤距离 dist 越大 → 解越稀疏、越珍贵 → 越要保留
# 输入一组帕累托解，计算每个解周围 “挤不挤”
def crowding_distance(solutions):
    n = len(solutions) # 解的个数
    if n == 0:
        return []
    f1_list = [f1(x) for x in solutions]  # 把所有解的 f1 值放进列表
    f2_list = [f2(x) for x in solutions]  # 把所有解的 f2 值放进列表
    f1_min, f1_max = min(f1_list), max(f1_list) # 算出 f1 的最小值、最大值
    f2_min, f2_max = min(f2_list), max(f2_list) # 算出 f2 的最小值、最大值
    f1_range = f1_max - f1_min if f1_max != f1_min else 1  # 计算 f1 的取值范围
    f2_range = f2_max - f2_min if f2_max != f2_min else 1  # 计算 f2 的取值范围

    # np.argsort 返回数组从小到大排序后【对应元素的原始索引】，不会修改原数组本身。
    sorted_idx = np.argsort(f1_list)  # 按 f1 值从小到大排序后的解的编号
    dist = np.zeros(n)  # 创建全 0 数组，用来存放每个解的拥挤距离
    # 边界上的解非常重要（代表极值），设为无穷大，算法永远不会删掉它们，保证帕累托前沿的完整性
    dist[sorted_idx[0]] = np.inf  # 将f1值最小的那个解的拥挤距离设为无穷大
    dist[sorted_idx[-1]] = np.inf # 将f1值最大的那个解的拥挤距离设为无穷大

    for i in range(1, n - 1):  # i=1,2,...,n-2
        dist[sorted_idx[i]] = (abs(f1_list[sorted_idx[i + 1]] - f1_list[sorted_idx[i - 1]]) / f1_range +
                               abs(f2_list[sorted_idx[i + 1]] - f2_list[sorted_idx[i - 1]]) / f2_range)
    return dist  # 返回每个解的拥挤距离

# 更新Pareto外部存档（去支配+去重+拥挤修剪）
def update_archive(archive, pop):
    combined = archive + pop
    non_dominated = [] # 新建空列表，存放帕累托最优解，即非支配解
    #对每个解 x，用y去检查是否支配x；一旦发现有y支配x，立刻执行break不再比；全程比完没有能支配x，x就是优秀解，进入存档
    for x in combined:
        dominated = False
        for y in combined:
            if is_dominates(y, x): # y支配x，条件成立，执行下面 2句
                dominated = True
                break  # 属于if is_dominates(y, x)这个语句块的语句
        if not dominated:
            non_dominated.append(x)

    unique = [] # 创建一个空列表，用来存放【去重后的最优解】
    for s in non_dominated:  #遍历所有非支配解（前面筛选出来的优秀解）
        if not any(np.array_equal(s, u) for u in unique): # 2个列表/数组全部相同 → array_equal返回 True
            unique.append(s)

    if len(unique) > N:
        dist = crowding_distance(unique)
        sorted_idx = np.argsort(dist)[::-1]  # 把解按照【拥挤距离从小到大】排序，得到对应解的下标，然后再反转顺序
        unique = [unique[i] for i in sorted_idx[:N]] # 只取前 N 个最稀疏、最珍贵的解，剩下的全部丢掉

    return unique  # 返回不重复的非支配解集合

# 初始化存档
A = update_archive([], Pop)

# ==============================================
# 10. 【主迭代循环】IMOGWO-SA/D
# ==============================================
iter = 0
T = T0

print("=====================================")
print("  IMOGWO-SA/D 运行中 ")
print("=====================================")

while iter < MaxIter and T > T_end:
    if iter % 10 == 0 and iter > 0:  # 每 10 代执行一次，扰动10%个体，第 0 代不执行
        perturb_num = int(M * 0.1)   # 随机换掉 10% 的解
        perturb_idx = random.sample(range(M), perturb_num) # 随机选出 10% 的个体编号，准备替换
        for idx in perturb_idx:
            x, st = init_solution()  # 随机生成一个全新的解
            Pop[idx] = x  # 把种群里第 idx 号旧解，直接替换成新解 x
            StartTimes[idx] = st  # 同步更新对应位置的开始时间

    # 遍历每个子问题 k
    for k in range(M):  # 例如 k=0..99
        S_N = [Pop[idx] for idx in neighborhood[k]] # 取出邻域中的解
        scores = [g(k, x) for x in S_N]  # 求出每个邻域解的切比雪夫函数值
        sorted_idx = np.argsort(scores)  # 获取按照g值从小到大排序后的邻域解的编号顺序

        # 三只头狼 α, β, δ
        alpha = S_N[sorted_idx[0]]  # 最优解：子问题k的g值最小的那个邻域解
        beta = S_N[sorted_idx[1]]   # 第二优
        delta = S_N[sorted_idx[2]]  # 第三优

        # 新建新解
        X_new = Pop[k].copy()
        st_new = StartTimes[k].copy()

        # 自适应权重（前期探索，后期收敛）
        # w_alpha = max(0.1, 2 - 1.9 * iter / MaxIter)
        w_alpha = max(0.1, 1.5 - iter / MaxIter)
        w_beta = max(0.1, w_alpha - 0.4)
        w_delta = max(0.1, w_beta - 0.4)

        # 已分配任务的农户集合
        assigned_j = set(np.argmax(X_new, axis=1)) # 所有已经被分配了任务、不能再接单的农户编号

        # 逐个任务更新
        for i in range(m):
            ja = np.argmax(alpha[i]) # 头狼 α 给第 i 个任务推荐的 分配农户编号
            jb = np.argmax(beta[i])  # 次狼 β 给第 i 个任务推荐的 分配农户编号
            jd = np.argmax(delta[i]) # 三狼 δ 给第 i 个任务推荐的 分配农户编号

            # 找当前任务i的可用农户
            F_i = []   # 存放当前任务i的可用农户
            for j in range(n):
                if j not in assigned_j and is_able(i, j):
                    s = max(farmers[j]["wf_s"], tasks[i]["wt_s"])
                    if s + t_ij[i][j] <= min(farmers[j]["wf_e"], tasks[i]["wt_e"]):
                        F_i.append((j, s))
            if not F_i:
                continue

            # 计算农户被选择的概率
            prob = []  # 存放可选农户的概率
            for j, s in F_i:
                pa = w_alpha * (1 if j == ja else 0)
                pb = w_beta * (1 if j == jb else 0)
                pd = w_delta * (1 if j == jd else 0)
                pr = omega_rand / len(F_i)
                prob.append(max(0.0, pa + pb + pd + pr))

            prob = np.array(prob) # 转成numpy数组
            s_prob = prob.sum()   # 计算概率总和
            if s_prob < 1e-8:     # 如果总和几乎为0（异常情况）
                # np.ones_like(prob)生成一个和 prob 长度一样、所有元素都是 1 的数组
                # 强制让每个农户被选中的机会完全相同，纯随机选一个
                prob = np.ones_like(prob) / len(prob)
            else:
                prob /= s_prob  # 把可用农户的概率归一化，让总和 = 1

            # 按概率选择农户，概率大 → 更容易被选择
            chosen = np.random.choice(len(F_i), p=prob)  # 被选中农户在 F_i 里的位置编号
            j_sel, s_sel = F_i[chosen]  # 把选上农户的编号和开始时间保存起来

            # 更新当前任务i的分配关系
            old_j = np.argmax(X_new[i])
            X_new[i] = 0
            X_new[i][j_sel] = 1       # 把当前任务i分配给新选中的农户
            st_new[i][j_sel] = s_sel  # 更新开始时间矩阵
            assigned_j.remove(old_j)  # 释放旧的农户
            assigned_j.add(j_sel)     # 将新选择的农户加入已分配任务的农户列表

        # =====================
        # SA 模拟退火扰动
        # =====================
        X_pert = X_new.copy()
        st_pert = st_new.copy()

        if m >= 2:
            i1, i2 = random.sample(range(m), 2)  # 随机选出 2 个不同的任务编号
            j1 = np.argmax(X_pert[i1])  # 找到任务 i1 当前分配的农户
            j2 = np.argmax(X_pert[i2])  # 找到任务 i2 当前分配的农户

            s1 = max(farmers[j2]["wf_s"], tasks[i1]["wt_s"])
            s2 = max(farmers[j1]["wf_s"], tasks[i2]["wt_s"])

            if (s1 + t_ij[i1][j2] <= min(farmers[j2]["wf_e"], tasks[i1]["wt_e"]) and
                    s2 + t_ij[i2][j1] <= min(farmers[j1]["wf_e"], tasks[i2]["wt_e"])):
                X_pert[i1] = 0
                X_pert[i1][j2] = 1
                X_pert[i2] = 0
                X_pert[i2][j1] = 1
                st_pert[i1][j2] = s1  # 更新开始时间矩阵
                st_pert[i2][j1] = s2  # 更新开始时间矩阵

        # Metropolis 接受准则
        dg = g(k, X_pert) - g(k, X_new)
        # random.random()生成一个 0 ~ 1 之间的随机小数
        if dg < 0 or random.random() < np.exp(-dg / T): # 好的解一定留；差的解，有概率留，帮助算法跳出局部最优！
            X_new = X_pert
            st_new = st_pert

        # 非法解修复
        if not is_feasible(X_new, st_new):  # 综合约束判断：解是否合法
            X_new, st_new = init_solution()

        # 更新种群
        Pop[k] = X_new
        StartTimes[k] = st_new

    # =====================
    # 更新外部存档与理想点
    # =====================
    A = update_archive(A, Pop)
    z_star[0] = min(f1(x) for x in A)
    z_star[1] = min(f2(x) for x in A)

    # 降温
    T *= alpha_cool
    iter += 1
    print(f"Iter {iter:3d} | Pareto解数量: {len(A):2d} | 温度: {T:6.2f}")

# ==============================================
# 11. 输出结果 + 绘制Pareto图
# ==============================================
print("\n" + "=" * 60)
print("                最终 Pareto 最优解               ")
print("=" * 60)

cost_list = []
profit_list = []
for idx, sol in enumerate(A):  # enumerate(A) 输出 (编号，元素)
    cost_val = f2(sol)
    profit_val = -f1(sol)
    cost_list.append(cost_val)
    profit_list.append(profit_val)
    print(f"\n--- 解 {idx + 1} ---")
    print(f"平台收益: {profit_val:.2f} | 农户成本: {cost_val:.2f}")
    for i in range(m):
        j = np.argmax(sol[i])
        print(f"任务{i} → 农户{j}")

# 绘图设置（中文正常显示）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# x轴=效益, y轴=成本
# ==============================
sorted_pairs = sorted(zip(profit_list, cost_list))
sorted_profit, sorted_cost = zip(*sorted_pairs)
plt.figure(figsize=(10, 6))
plt.scatter(sorted_profit, sorted_cost, c='red', s=80, label='Pareto 最优解')  # 画散点图
plt.plot(sorted_profit, sorted_cost, 'b--', linewidth=2, label='Pareto 前沿')  # 画一条线
plt.xlabel('平台总收益', fontsize=12)  # 横坐标为收益
plt.ylabel('农户总成本', fontsize=12)  # 纵坐标为成本
plt.title('IMOGWO-SA/D Pareto 前沿图', fontsize=14)
plt.legend()  # 给图表加上图例
plt.grid(True)
plt.savefig('Pareto_front_A4.png', dpi=300, bbox_inches='tight') # bbox_inches='tight'：自动裁剪白边
plt.show()