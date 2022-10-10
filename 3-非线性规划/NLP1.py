import numpy as np
import sympy as sym
from scipy.optimize import minimize

def objective_function():
    obj = lambda x: -(3*x[0] - (x[0]-1)**2 + 3*x[1] - (x[1]-2)**2)
    return obj

def constrains():
    cons = ({'type': 'ineq', 'fun': lambda x: 4 * x[0] + x[1] - 20},
        {'type': 'ineq', 'fun': lambda x: 4 * x[1] + x[0] - 20})
    return cons

if __name__ == "__main__":
    ## 初始迭代点
    x0 = np.array([1,1])
    ## 约束
    cons = constrains()
    bnds = ((0,None),(0,None))
    res = minimize(objective_function(), x0, bounds=bnds, constraints=cons)
    print(res)