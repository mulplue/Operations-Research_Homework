from pulp import *
import numpy as np

## Data
m = 2                           # m种炸弹(i)
n = 4                           # n个要害(j)

p = [
        [0.1,0.2,0.15,0.25],
        [0.08,0.16,0.12,0.20]
    ]                           # 第i种炸弹轰炸要害j成功的可能性(p_ij)

c = [
        [537.5,560,605,650],
        [462.5,480,515,550]
    ]                           # 第i种炸弹轰炸要害j花费的油量(c_ij)


## Problem

problem = LpProblem('Bomber_Problem', LpMinimize)
## Variables

x = [ [LpVariable("x"+str(i+1)+str(j+1),lowBound=0,cat=const.LpInteger) for j in range(n)] for i in range(m)]


## Objective Function
obj = lpSum(np.log(1-p[i][j])*x[i][j] for j in range(n) for i in range(m))
problem += obj, 'Objective Function'



## Constraints
problem += lpSum(x[0][j] for j in range(n)) <= 48 ,'Big_Bomb_Number_Constraints'
problem += lpSum(x[1][j] for j in range(n)) <= 32 ,'Small_Bomb_Number_Constraints'
problem += lpSum(x[i][j]*c[i][j] for j in range(n) for i in range(m)) <= 48000  ,'Oil_Constraints'
## Solve
print(problem)
problem.solve()

## Print
for i in range(m):
    for j in range(n):
        print(x[i][j].varValue)