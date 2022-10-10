import numpy as np
from pulp import *
import queue

if __name__ == "__main__":

    ## Problem
    prob = LpProblem('Test', LpMaximize)
    ## Variables
    eps = np.finfo(np.float64).eps
    x1 = LpVariable("x1",lowBound=0+eps)
    x2 = LpVariable("x2",lowBound=0+eps)
    x = [x1, x2]
    ## Objective Function
    obj = 40*x1 + 90*x2
    prob += obj, 'Objective Function'
    ## Fixed Constraints
    prob += 9*x1 + 7*x2 <= 56
    prob += 7*x1 + 20*x2 <= 70


    ## Branch and Bound
    # 初始化
    low_bound = 0           # 初始下界,一定大于0
    opt_max = None
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
            """注意这行代码注释与否的效果"""
            # prob_now.solve()

            # print("after queue:")
            # for v in prob_now.variables():
            #     print(value(v))
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
                # print("before queue:")
                # for v in prob_low.variables():
                #     print(value(v))
                prob_up.solve()

                # 加队列
                if prob_low.status == 1:
                    Q.put(prob_low)
                if prob_up.status == 1:
                    Q.put(prob_up)

        return best_x, om

    x,obj = _BB(Q,low_bound,opt_max, best_x)
    print("x:",x)
    print("obj:",obj)