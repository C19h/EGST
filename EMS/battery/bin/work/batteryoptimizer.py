from scipy.optimize import minimize

class BatteryOptimizer(object):
    def __init__(self, T):
        self.T = T

    def set_params(self, uph, gph, bpph, spph, blph, slph, bco, socmax, socmin, bum, bcm, ptype = 0):
        """
        | T                 |  N      | int     | 时间周期，如24代表将1天划分为24份，每个点代表一个小时；48代表将1天划分为48份，每个点代表半个小时。默认为24 |
        | UsagePerHour      |  Y      | float[] | 下一天每小时用电量预测（如果T为48则代表每半小时，数组长度应对应T，下同） |
        | GeneratePerHour   |  Y      | float[] | 下一天每小时发电量预测（负为发电，正为用电）|
        | BuyPricePerHour   |  Y      | float[] | 下一天每小时买电电价 |
        | SellPricePerHour  |  Y      | float[] | 下一天每小时卖电电价 |
        | BuyLimitPerHour   |  Y      | float[] | 下一天每小时买电上限（为正数） |
        | SellLimitPerHour  |  Y      | float[] | 下一天每小时卖电上限（为正数） |
        | BatteryCost       |  Y      | float   | 电池单位功率的成本 |
        | SOCMax            |  Y      | float   | 电池可用容量上限 |
        | SOCMin            |  Y      | float   | 电池可用容量下限 |
        | BatteryUsageMax   |  Y      | float   | 电池放电功率上限（为正数） |
        | BatteryChargeMax  |  Y      | float   | 电池充电功率上限（为正数） |
        | PlanType          |  Y      | int     | 0为计算每小时买卖电功率上限<br>1为不限制功率上限（SLimit、BLimit无效） |
        """
        if not isinstance(bpph, list):
            bpph = [bpph] * self.T
        if not isinstance(spph, list):
            spph = [spph] * self.T
        if not isinstance(blph, list):
            blph = [blph] * self.T
        if not isinstance(slph, list):
            slph = [slph] * self.T
        self.UsagePerHour = uph
        self.GeneratePerHour = gph
        self.BuyPricePerHour = bpph
        self.SellPricePerHour = spph
        self.BuyLimitPerHour = blph
        self.SellLimitPerHour = slph
        self.BatteryCostPerPower = bco
        self.SOCMax = socmax
        self.SOCMin = socmin
        self.BatteryUsageMax = bum
        self.BatteryChargeMax = bcm
        self.PlanType = ptype

    def calc_plan(self):
        x0 = [0] * self.T
        #约束1，电池每天结束后充放平衡
        constraints = []
        constraints.append({
                'type': 'eq', 
                'fun': lambda x: sum(x)
            })
        #约束2，Soc每个小时都不能越界
        for i in range(self.T):
            constraints.append({
                'type': 'ineq', 
                'args': (i,), 
                'fun': lambda x,i: (self.SOCMax - sum(x[:i+1])) - self.SOCMin
            })
            constraints.append({
                'type': 'ineq', 
                'args': (i,), 
                'fun': lambda x,i: sum(x[:i+1])
            })
        #约束3，每个小时从电网买卖电限额
        if self.PlanType == 0:
            for i in range(self.T):
                constraints.append({
                    'type': 'ineq', 
                    'args': (i,), 
                    'fun': lambda x,i: (self.UsagePerHour[i] + self.GeneratePerHour[i] - x[i]) + self.SellLimitPerHour[i]
                })
                constraints.append({
                    'type': 'ineq', 
                    'args': (i,), 
                    'fun': lambda x,i: self.BuyLimitPerHour[i] - (self.UsagePerHour[i] + self.GeneratePerHour[i] - x[i])
                })
        #约束4，边界
        bound = []
        for i in range(self.T):
            bound.append((self.BatteryChargeMax*-1, self.BatteryUsageMax))
        #求解
        res = minimize(self._calc_cost, x0, method='SLSQP', constraints=constraints, bounds = bound)
        ret = self._make_return(res)
        return ret
    
    def calc_plan_in_day(self, x1):
        LT = len(x1)
        NT = self.T - LT
        sumlastx = sum(x1[:])
        x0 = [0] * NT
        #约束1，电池每天结束后充放平衡
        constraints = []
        constraints.append({
                'type': 'eq', 
                'fun': lambda x: sum(x) + sumlastx
            })
        #约束2，Soc每个小时都不能越界
        for i in range(NT):
            constraints.append({
                'type': 'ineq', 
                'args': (i,), 
                'fun': lambda x,i: (self.SOCMax - sum(x[:i+1]) - sumlastx) - self.SOCMin
            })
            constraints.append({
                'type': 'ineq', 
                'args': (i,), 
                'fun': lambda x,i: sum(x[:i+1]) + sumlastx
            })
        #约束3，每个小时从电网买卖电限额
        if self.PlanType == 0:
            for i in range(NT):
                constraints.append({
                    'type': 'ineq', 
                    'args': (i,), 
                    'fun': lambda x,i: (self.UsagePerHour[LT + i] + self.GeneratePerHour[LT + i] - x[i]) + self.SellLimitPerHour[LT + i]
                })
                constraints.append({
                    'type': 'ineq', 
                    'args': (i,), 
                    'fun': lambda x,i: self.BuyLimitPerHour[LT + i] - (self.UsagePerHour[LT + i] + self.GeneratePerHour[LT + i] - x[i])
                })
        #约束4，边界
        bound = []
        for i in range(NT):
            bound.append((self.BatteryChargeMax*-1, self.BatteryUsageMax))
        res = minimize(self._calc_cost, x0, method='SLSQP', constraints=constraints, bounds = bound)
        ret = self._make_return(res, x1)
        return ret

    def _make_return(self, res, x0 = None):
        print('最小值：',res.fun)
        print('最优解：',res.x)
        print('迭代终止是否成功：', res.success)
        print('迭代终止原因：', res.message)
        if not x0 is None:
            x = x0
            x.extend(res.x)
        else:
            x = res.x
        ret = {}
        ret['Success'] = 1 if res.success == True else 0
        ret['SOCPerHour'] = []
        ret['BatteryPlanPerHour'] = []
        ret['ShortagePerHour'] = []
        ret['PlanBuyPerHour'] = []
        ret['PlanCostPerHour'] = []
        for i in range(self.T):
            ret['SOCPerHour'].append(round(self.SOCMax - sum(x[:i+1]),2))
            ret['BatteryPlanPerHour'].append(round(x[i],2))
            ret['ShortagePerHour'].append(round(self.UsagePerHour[i] + self.GeneratePerHour[i],2))
            ret['PlanBuyPerHour'].append(round(self.UsagePerHour[i] + self.GeneratePerHour[i] - x[i],2))
            ret['PlanCostPerHour'].append(round(self._calc_cost_per_hour(0, x, i),2))
        ret['TotalCost'] = round(sum(ret['PlanCostPerHour']),2)
        return ret

    def _calc_cost(self, x):
        cost = 0
        starthour = self.T - len(x)
        for i in range(len(x)):
            cost += self._calc_cost_per_hour(starthour, x, i)
        return cost

    def _calc_cost_per_hour(self, starthour, x, i):
        #电池成本
        cost = abs(x[i]) * self.BatteryCostPerPower
        gp = self.UsagePerHour[starthour + i] + self.GeneratePerHour[starthour + i] - x[i]
        if gp > 0:  #买电
            cost += gp * self.BuyPricePerHour[starthour + i]
        else:       #卖电
            cost += gp * self.SellPricePerHour[starthour + i] 
        return cost