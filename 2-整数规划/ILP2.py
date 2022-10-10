from pulp import *
import numpy as np

## Data
n = 5                           # n个物品(j)
a = [1,2,3,4,5]                 # 第j个物品的质量(a_j)
c = [2,4,4,5,6]                 # 第j个物品的价值(c_j)

b = 6                           # 背包容积

## Problem
problem = LpProblem('Knapsack_Problem', LpMaximize)
## Variables
x = [LpVariable('x'+str(j+1),cat=LpBinary) for j in range(n)]
## Objective Function
obj = lpSum(c[j]*x[j] for j in range(n))
problem += obj, 'Objective Function'
## Constraints
problem += lpSum(x[j]*a[j] for j in range(n)) <= b

## Solve
print(problem)
problem.solve()

## Print
for v in problem.variables():
    print(value(v))