from pulp import *

## Data
m = 2                           # m种营养成分(i)
n = 3                           # n种食品(j)
b = [None,3500,2340]            # 第i种营养的每日必需量(b_i)
c = [None,0.18,0.23,0.05]       # 第j种食品的cost(c_j)
d = [None,5,3,7]                # 第j种食品的每日必需量(d_j)
a = [   None,
        [None,107,500,0],
        [None,72,121,65]
    ]                           # 第j种食品含有的i类成分(a_ij)


## Problem
problem = LpProblem('Diet_Problem', LpMinimize)
## Variables
x = [None]
for j in range(1,n+1):
    xj = LpVariable("x"+str(j),lowBound=d[j])
    x.append(xj)

## Objective Function
obj = 0
for j in range(1,n+1):
    obj += c[j]*x[j]
problem += obj, 'Objective Function'

## Constraints
for i in range(1,m+1):
    cons1i = lpSum(a[i][j]*x[j] for j in range(1,n+1))
    problem += cons1i >= b[i] , 'Nutrition_Constraints'+str(i)

## Solve
print(problem)
problem.solve()

## Print
for i in range(1,n+1):
    print("x"+str(i)+":",x[i].varValue)