import numpy as np

# 周期
T = 24
# 每小时用电
UPh = np.array([1, 1, 1, 1, 1, 1, 1, 1,  # 0-8
                10, 10, 10, 10,
                3, 8,
                10, 10, 10, 6,
                2, 2, 2, 2,
                1, 1], dtype=np.float)
LPh = np.array([0, 0, 0, 0, 0, 0, -1, -2,
                -3, -4, -4, -4, -4, -4, -3, -2, -1, 0,
                0, 0, 0, 0, 0, 0], dtype=np.float)  # 每小时光伏发电
BCost = 0.1  # 电池使用成本
# 每小时买电电价,0点起,单位元/千瓦时
BPh = np.array([0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45,  # 0-7,45%
                1,  # 7-8,100%
                1.34, 1.34, 1.34,  # 8-11,134%
                1, 1, 1, 1,  # 11-15,100%
                1.34, 1.34, 1.34, 1.34,
                1.8, 1.8, 1.8,
                1,
                0.45], dtype=np.float) * 0.5730
# 每小时卖电电价,0点起,单位元/千瓦时
SPh = np.array([0.69, 0.69, 0.69, 0.69, 0.69, 0.69, 0.69,  # 0-7,45%
                1,  # 7-8,100%
                1.17, 1.17, 1.17,  # 8-11,134%
                1, 1, 1, 1,  # 11-15,100%
                1.17, 1.17, 1.17, 1.17,
                1.17, 1.17, 1.17,
                1,
                0.69], dtype=np.float) * 0.2730
for i in range(T):
    SPh[i] = round(SPh[i], 2)
# 电网供电限制
BLimit = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], dtype=np.float)
# 电网卖电限制
SLimit = np.array([8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8], dtype=np.float)
SOCMax = 10
SOCMin = 1
BUMax = 4
BCMax = 3
###########################################################
from scipy.optimize import minimize


def calc(UPh, LPh, BCost, SPh, BPh, SLimit, BLimit, SOCMax, SOCMin, BUMax, BCMax):
    """
    :param UPh: 每小时用电
    :param LPh: 每小时光伏发电
    :param BCost: 电池使用成本
    :param SPh: 每小时卖电电价
    :param BPh: 每小时买电电价
    :param SLimit: 电网每小时卖电限制
    :param BLimit: 电网每小时供电限制
    :param SOCMax: SOC最大值
    :param SOCMin: SOC最小值
    :param BUMax: 4
    :param BCMax: 3
    :return:
    """
    T = 24
    if isinstance(SPh, float) or isinstance(SPh, int):
        SPh = [SPh] * T
    if isinstance(SLimit, float) or isinstance(SLimit, int):
        SLimit = [SLimit] * T
    if isinstance(BLimit, float) or isinstance(BLimit, int):
        BLimit = [BLimit] * T
    constraints = []
    # 约束1，电池每天结束后充放平衡
    constraints.append({
        'type': 'eq',
        'fun': lambda x: sum(x)
    })
    # 约束2，Soc每个小时都不能越界
    for i in range(T):
        constraints.append({
            'type': 'ineq',
            'args': (i,),
            'fun': lambda x, i: (SOCMax - sum(x[:i + 1])) - SOCMin
        })
        constraints.append({
            'type': 'ineq',
            'args': (i,),
            'fun': lambda x, i: sum(x[:i + 1])
        })
    # 约束3，每个小时从电网买卖电限额
    for i in range(T):
        constraints.append({
            'type': 'ineq',
            'args': (i,),
            'fun': lambda x, i: (UPh[i] + LPh[i] - x[i]) + SLimit[i]
        })
        constraints.append({
            'type': 'ineq',
            'args': (i,),
            'fun': lambda x, i: BLimit[i] - (UPh[i] + LPh[i] - x[i])
        })

    # 成本函数
    def calcCost(x):
        cost = 0
        for i in range(T):
            cost += calcCostPH(x, i)
        return cost

    def calcCostPH(x, i):
        # 电池成本
        cost = abs(x[i]) * BCost
        gp = UPh[i] + LPh[i] - x[i]
        if gp > 0:  # 买电
            cost += gp * BPh[i]
        else:  # 卖电
            cost += gp * SPh[i]
        return cost

    bound = []
    for i in range(T):
        bound.append((BCMax * -1, BUMax))
    x0 = [0] * T
    for i in range(T):
        x0[i] = UPh[i] + LPh[i]
    res = minimize(calcCost, x0, method='SLSQP', constraints=constraints, bounds=bound)
    print('最小值：', res.fun)
    print('最优解：', res.x)
    print('迭代终止是否成功：', res.success)
    print('迭代终止原因：', res.message)
    ret = {}
    ret['SOC'] = []
    ret['U'] = []
    ret['Need'] = []
    ret['Cost'] = []
    ret['Sell'] = []
    ret['L'] = []
    ret['Cost'] = []
    ret['Uph'] = []
    for i in range(T):
        ret['SOC'].append(round(SOCMax - sum(res.x[:i + 1]), 2))
        ret['U'].append(round(res.x[i], 2))
        ret['Need'].append(round(UPh[i] + LPh[i], 2))
        ret['Sell'].append(round(UPh[i] + LPh[i] - res.x[i], 2))
        ret['Cost'].append(round(calcCostPH(res.x, i), 2))
        ret['L'].append(round(LPh[i], 2))
        ret['Uph'].append(round(UPh[i], 2))
    ret['TotalCost'] = round(sum(ret['Cost']), 2)
    return ret


res = calc(UPh, LPh, BCost, SPh, BPh, SLimit, BLimit, SOCMax, SOCMin, BUMax, BCMax)
