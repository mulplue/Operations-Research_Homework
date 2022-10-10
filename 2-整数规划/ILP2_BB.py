from pulp import *
import numpy as np
import queue

## Data
n = 5                           # n个物品(j)
a = [1,2,3,4,5]                 # 第j个物品的质量(a_j)
c = [2,4,4,5,6]                 # 第j个物品的价值(c_j)

b = 6                           # 背包容积

## Problem
problem = LpProblem('Knapsack_Problem', LpMaximize)
## Variables
x = [LpVariable('x'+str(j+1),lowBound=0) for j in range(n)]
## Objective Function
obj = lpSum(c[j]*x[j] for j in range(n))
problem += obj, 'Objective Function'
## Constraints
problem += lpSum(x[j]*a[j] for j in range(n)) <= b
for j in range(n):
    problem += x[j]<=1


## Solve
# 初始化
low_bound = 0           # 初始下界,一定大于0
opt_max = None
best_x = None
Q = queue.Queue()
# 解初始问题
problem.solve()


if problem.status !=1:
    raise ValueError('Insoluble!')
else:
    Q.put(problem)

# 主递归流程
def _BB(Q, low_bound, opt_max, best_x):
    lb = low_bound
    om = opt_max
    # 循环遍历所有子问题
    while not Q.empty():
        prob_now = Q.get()
        """注意这行代码注释与否的效果"""
        prob_now.solve()
        obj_now = value(prob_now.objective)

        # 若该区域的最大值小于下界,则直接排除
        if obj_now < lb:
            continue

        # 遍历,寻找第一个非整数解
        flag = 1
        for v in prob_now.variables():
            tmp = value(v)
            if not value(v).is_integer():
                flag = 0
                break

        # 整数解,与下界比较更新
        if flag == 1:
            if lb < obj_now:
                lb = obj_now
            if om is None or obj_now > om:
                om = obj_now
                best_x = [value(v) for v in prob_now.variables()]

        # 非整数解,需要分枝
        else:
            branch_v = None
            for v in prob_now.variables():
                if not value(v).is_integer():
                    branch_v = v
                    break
            # 新约束
            side_low = np.floor(value(branch_v))
            side_up = np.ceil(value(branch_v))
            v_i = int(str(branch_v)[1])

            # 子问题
            prob_low = prob_now.deepcopy()
            prob_up = prob_now.deepcopy()
            prob_low += x[v_i-1] <= side_low
            prob_up += x[v_i-1] >= side_up

            # 求解
            prob_low.solve()
            prob_up.solve()

            # 加队列
            if prob_low.status == 1:
                Q.put(prob_low)
            if prob_up.status == 1:
                Q.put(prob_up)

    return best_x, om

x,obj = _BB(Q,low_bound,opt_max, best_x)
print("x:", x)
print("obj:", obj)