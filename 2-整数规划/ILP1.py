import numpy as np
from pulp import *
import queue

if __name__ == "__main__":
    ## data
    m, n = 5, 6                         # i上限m, j上限n
    r = [
        [4,3,4,4,5,6],
        [3,4,5,3,4,5],
        [5,3,4,5,5,4],
        [3,3,4,4,6,6],
        [3,3,3,4,5,7]
    ]                                   # 第i个车间生产第j种布料的利润(r_ij)

    a = [6,6,7,8,9,10]                  # 第j种布料的单价(a_j)

    fund = 4e5                          # 总资金
    cloth_min, cloth_max = 1e3, 1e4     # 加工上下限


    ## Problem
    prob = LpProblem('Cloth_Distribution', LpMaximize)
    ## Variables
    x = [ [LpVariable("x"+str(i+1)+str(j+1),lowBound=cloth_min) for j in range(n)] for i in range(m) ]
    ## Objective Function
    obj = lpSum(r[i][j]*x[i][j] for j in range(n) for i in range(m))
    prob += obj, 'Objective Function'
    ## Fixed Constraints
    prob += lpSum(x[i][j]*a[j] for i in range(m) for j in range(n)) <= fund
    for i in range(m):
        prob += lpSum(x[i][j] for j in range(n)) <= cloth_max


    ## Branch and Bound
    # 初始化
    low_bound = 0           # 初始下界,一定大于0
    best_x = None
    Q = queue.Queue()
    # 解初始问题
    prob.solve()
    if prob.status !=1:
        raise ValueError('Insoluble!')      # 无可行解
    else:
        Q.put(prob)                         # 可解,放入队列

    # 主递归流程
    def _BB(Q, low_bound, opt_max, best_x):
        lb = low_bound
        om = opt_max
        # 循环遍历所有子问题
        while not Q.empty():
            prob_now = Q.get()
            prob_now.solve()
            obj_now = value(prob_now.objective)

            # 若该区域的最大值小于下界,则直接排除
            if obj_now < lb:
                continue

            # 遍历,寻找第一个非整数解
            flag = 1
            for v in prob_now.variables():
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
                v_i, v_j = int(str(branch_v)[1]), int(str(branch_v)[2])

                # 子问题
                prob_low = prob_now.deepcopy()
                prob_up = prob_now.deepcopy()
                prob_low += x[v_i-1][v_j-1] <= side_low
                prob_up += x[v_i-1][v_j-1] >= side_up

                # 求解
                prob_low.solve()
                prob_up.solve()

                # 加队列
                if prob_low.status == 1:
                    Q.put(prob_low)
                if prob_up.status == 1:
                    Q.put(prob_up)

        return best_x, om

    x,obj = _BB(Q,low_bound,up_bound, best_x)
    print("x:",x)
    print("obj:",obj)