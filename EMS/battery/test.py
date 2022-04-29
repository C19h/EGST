import numpy as np
#周期
T = 24
#每小时用电
Pb = np.array([1,1,1,1,1,1,1,1,                      #0-8
    10,10,10,10,
    4,8,
    10,10,10,6,
    2,2,2,2,
    1,1], dtype=np.float)
Ppv = np.array([0,0,0,0,0,0,-1,-2,
    -3,-4,-4,-4,-4,-4,-3,-2,-1,0,
    0,0,0,0,0,0], dtype=np.float)     #每小时光伏发电
Bc = 0.1    #电池使用成本
#每小时买电电价,0点起,单位元/千瓦时
Bph = np.array([0.45,0.45,0.45,0.45,0.45,0.45,0.45,  #0-7,45%
    1,                                      #7-8,100%
    1.34,1.34,1.34,                         #8-11,134%
    1,1,1,1,                                #11-15,100%
    1.34,1.34,1.34,1.34,
    1.8,1.8,1.8,
    1,
    0.45], dtype=np.float) * 0.5730
#每小时卖电电价,0点起,单位元/千瓦时 
Sph = np.array([0.69,0.69,0.69,0.69,0.69,0.69,0.69,  #0-7,45%
    1,                                      #7-8,100%
    1.17,1.17,1.17,                         #8-11,134%
    1,1,1,1,                                #11-15,100%
    1.17,1.17,1.17,1.17,
    1.17,1.17,1.17,
    1,
    0.69], dtype=np.float) * 0.2730
#电网供电限制
GMax = np.array([8,8,8,8,8,8,8,8,8,8,8,8,  
    8,8,8,8,8,8,8,8,8,8,8,8], dtype=np.float)
#电网卖电限制
GMin = np.array([-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,  
    -8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8,-8], dtype=np.float)
SocMax = 10
SocMin = 1
BMax = 4
BMin = -3
###########################################################
import pulp
#电量缺口
ENeed = Pb + Ppv

problem = pulp.LpProblem("My_LP_Problem", pulp.LpMinimize)
x = []
for i in range(T):
    temp = pulp.LpVariable('x%d' % i, lowBound=BMin, upBound=BMax, cat='Continuous')
    x.append(temp)
#约束1，电池每天结束后充放平衡
temp = x[0]
for i in range(1,T):
    temp += x[i]
problem += temp == 0
#约束2，Soc每个小时都不能越界
for i in range(T):
    temp = x[0]
    for j in range(1,i + 1):
        temp += x[j]
    problem += temp >= SocMin
    problem += temp <= SocMax
#约束3，每个小时从电网买卖电限额
for i in range(T):
    temp = Pb[i] + Ppv[i] + x[i]
    problem += temp >= GMin[i]
    problem += temp <= GMax[i]
#成本函数
# 电池使用成本
def calcCost(xrow):
    cost = 0
    if xrow >= 0:
        cost = xrow * xrow * Bc
    else:
        cost = xrow * xrow * Bc
    return cost
problem += pulp.lpSum([calcCost(xrow) for xrow in x])
problem.solve()

for xrow in x:
    print('x:',xrow.value())


