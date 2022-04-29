import os,io,base64,json, sys, traceback
from flask import Blueprint, render_template, redirect, send_from_directory, request, Response
from logmanager import LogManager
from configmanager import ConfigManager
from batteryoptimizer import BatteryOptimizer

blue = Blueprint('battery',__name__)

@blue.route('/battery/calcPlan',methods=['post','get'])
def calc_plan():
    """计算峰谷套利
    ### 说明
    使用给出的下一日的用户用电量、光伏发电量等预测数据计算峰谷套利。该接口返回数据保留两位小数。
    <br>由于要保证预测的最后一个点使得电池重新充满，因此通常最后一个点代表最低买电电价的最后一个时间段，如06:00-07:00。
    <br>时间单位为小时，其它单位如价格、电量，使用统一单位即可，如价格统一使用元，电量单位统一使用kWh。
    <br>请注意算法中不涉及功率，均为每小时或每半小时的用电量与发电量。

    ### 请求参数
    |      参数名        | 是否必填 |  类型    | 说明      |
    |-------------------|:-------:|:-------:|----------|
    | T                 |  N      | int     | 时间周期，如24代表将1天划分为24份，每个点代表一个小时；48代表将1天划分为48份，每个点代表半个小时。默认为24。 |
    | UsagePerHour      |  Y      | float[] | 下一天每小时用电量预测（如果T为48则代表每半小时，数组长度应对应T，下同） |
    | GeneratePerHour   |  Y      | float[] | 下一天每小时发电量预测（负为发电，正为用电）|
    | BuyPricePerHour   |  Y      | float[] | 下一天每小时买电电价 |
    | SellPricePerHour  |  Y      | float[] | 下一天每小时卖电电价 |
    | BuyLimitPerHour   |  Y      | float[] | 下一天每小时买电上限（为正数） |
    | SellLimitPerHour  |  Y      | float[] | 下一天每小时卖电上限（为正数） |
    | BatteryCostPerPower |  Y      | float   | 电池单位电量的成本 |
    | SOCMax            |  Y      | float   | 电池可用容量上限 |
    | SOCMin            |  Y      | float   | 电池可用容量下限 |
    | BatteryUsageMax   |  Y      | float   | 电池每小时放电量上限（为正数） |
    | BatteryChargeMax  |  Y      | float   | 电池每小时充电量上限（为正数） |
    | PlanType          |  Y      | int     | 0为计算每小时买卖电量上限<br>1为不限制买卖电量上限（SLimit、BLimit无效） |

    ### 返回参数
    |    参数名  |  类型    |  说明 |
    |-----------|:-------:|-------|
    | code      |  int   | 状态码    |
    | success   |  int   | 是否成功  |
    | message   | string | 失败时的错误信息 |
    | data      | object | 返回数据 |
    | 1 Success         |  int    | 1为规划成功<br>0为规划失败，条件冲突 |
    | 2 ShortagePerHour | float[] | 每小时用电量缺口，为用户用电量减去光伏发电量（负代表可用于卖电） |
    | 3 PlanBuyPerHour  | float[] | 规划的每小时买电量（负为卖电量） |
    | 4 BatteryPlanPerHour | float[] | 规划的每小时电池放电量（正为放电，负为充电） |
    | 5 SOCPerHour      | float[] | 规划的每小时截止时的SOC剩余容量 |
    | 6 PlanCostPerHour | float[] | 规划的每小时的成本（负代表减去电池成本后的卖电收益） |
    | 7 TotalCost       | float   | 规划的总成本（成本为买电费用+电池使用成本） |
  
    ### 请求示例
    ```json
    {
        "T":24,
        "UsagePerHour":[100,100,100,120,130,120,110,100,120,140,160,170,110,100,90,80,70,60,50,40,30,20,10,10],
        "GeneratePerHour":[0,0,0,10,20,30,40,40,30,20,10,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "BuyPricePerHour":[0.57,0.77,0.77,0.77,0.57,0.57,0.57,0.57,0.77,0.77,0.77,0.77,1.03,1.03,1.03,0.57,0.26,0.26,0.26,0.26,0.26,0.26,0.26,0.26],
        "SellPricePerHour":[0.27,0.32,0.32,0.32,0.27,0.27,0.27,0.27,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.27,0.19,0.19,0.19,0.19,0.19,0.19,0.19,0.19],
        "SellLimitPerHour":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "BuyLimitPerHour":[800,800,800,200,200,200,200,200,200,120,120,120,800,800,800,800,800,800,800,800,800,800,800,800],
        "BatteryCostPerPower":0.1,
        "SOCMax":200,
        "SOCMin":20,
        "BatteryUsageMax":60,
        "BatteryChargeMax":50,
        "PlanType":0
    }
    ```

    ### 返回示例
    ```json
    {
        "data":{
            "Success":1,
            "SOCPerHour":[199.76,197.15,194.55,191.95,193.96,195.97,197.99,200,200,160,110,60,46.67,33.33,20,22.03,44.28,66.52,88.77,111.02,133.26,155.51,177.75,200],
            "BatteryPlanPerHour":[0.24,2.6,2.6,2.6,-2.01,-2.01,-2.01,-2.01,0,40,50,50,13.33,13.33,13.33,-2.03,-22.25,-22.25,-22.25,-22.25,-22.25,-22.25,-22.25,-22.25],
            "ShortagePerHour":[100,100,100,130,150,150,150,140,150,160,170,170,110,100,90,80,70,60,50,40,30,20,10,10],
            "PlanBuyPerHour":[99.76,97.4,97.4,127.4,152.01,152.01,152.01,142.01,150,120,120,120,96.67,86.67,76.67,82.03,92.25,82.25,72.25,62.25,52.25,42.25,32.25,32.25],
            "PlanCostPerHour":[56.89,75.26,75.26,98.36,86.85,86.85,86.85,81.15,115.5,96.4,97.4,97.4,100.9,90.6,80.3,46.96,26.21,23.61,21.01,18.41,15.81,13.21,10.61,10.61],
            "TotalCost":1512.41
        },
        "success":1,
        "message":"",
        "code":200
    }
    ```
    """
    try:
        jdata = json.loads(request.get_data().decode("utf-8"))
        uph  = jdata.get('UsagePerHour')
        gph  = jdata.get('GeneratePerHour')
        bpph = jdata.get('BuyPricePerHour')
        spph = jdata.get('SellPricePerHour')
        blph = jdata.get('BuyLimitPerHour')
        slph = jdata.get('SellLimitPerHour')
        bcpp  = float(jdata.get('BatteryCostPerPower'))
        socmax = float(jdata.get('SOCMax'))
        socmin = float(jdata.get('SOCMin'))
        bum  = float(jdata.get('BatteryUsageMax'))
        bcm  = float(jdata.get('BatteryChargeMax'))
        ptype = int(jdata.get('PlanType'))
        T = jdata.get('T')
        if T is None:
            model = BatteryOptimizer(24)
        else:
            model = BatteryOptimizer(int(T))
        model.set_params(uph, gph, bpph, spph, blph, slph, bcpp, socmax, socmin, bum, bcm, ptype)
        ret = model.calc_plan()
        return json.dumps({'data':ret,"success":1, "message":"","code":200}, ensure_ascii=False)
    except:
        msg = traceback.format_exc()
        LogManager.Instance.Log(__name__, 2, 'calc_plan error\n' + msg)
        return json.dumps({'data':msg,"success":0, "message":"查询失败","code":403}, ensure_ascii=False)

"""
日内预测
UPh：下一天每小时用电量预测（数组）
LPh：下一天每小时光伏电量预测（数组）
SPh：下一天每小时卖电电价（数组）
BPh：下一天每小时买电电价（数组）
SLimit：每小时卖电上限
BLimit：每小时买电上限
BCost：电池单位功率的成本
SOCMax：电池可用容量上限
SOCMin：电池可用容量下限
BUMax：电池每小时放电上限
BCMax：电池每小时充电上限
Type：0为计算每小时买卖电功率上限，1为不限制功率上限
Hour：实际已经经过的小时数，例如为2，那么UPh、LPh的第1、2个点，则应为本日实际第1、2个小时的用电、发电
Battery：
"""
@blue.route('/battery/calcPlanInDay',methods=['post','get'])
def calc_plan_in_day():
    """计算日内峰谷套利
    ### 说明
    根据当日已使用电池数据计算当日剩下时间的峰谷套利。返回数据仍包含当日已经过时间的数据。该接口返回数据保留两位小数。
    <br>由于要保证预测的最后一个点使得电池重新充满，因此通常最后一个点代表最低买电电价的最后一个时间段，如06:00-07:00。
    <br>时间单位为小时，其它单位如价格、电量，使用统一单位即可，如价格统一使用元，电量单位统一使用kWh。
    <br>请注意算法中不涉及功率，均为每小时或每半小时的用电量与发电量。

    ### 请求参数
    |      参数名        | 是否必填 |  类型    | 说明      |
    |-------------------|:-------:|:-------:|----------|
    | T                 |  N      | int     | 时间周期，如24代表将1天划分为24份，每个点代表一个小时；48代表将1天划分为48份，每个点代表半个小时。默认为24。 |
    | BatteryPerHour    |  Y      | float[] | 本日已过时间的每小时电池放电量，数组长度代表已过小时数（正为放电，负为充电）。如数组长度为3，则下列各预测与限制数据的前3个点对应过去的3小时，不再生效。 |
    | UsagePerHour      |  Y      | float[] | 下一天每小时用电量预测（如果T为48则代表每半小时，数组长度应对应T，下同） |
    | GeneratePerHour   |  Y      | float[] | 下一天每小时发电量预测（负为发电，正为用电）|
    | BuyPricePerHour   |  Y      | float[] | 下一天每小时买电电价 |
    | SellPricePerHour  |  Y      | float[] | 下一天每小时卖电电价 |
    | BuyLimitPerHour   |  Y      | float[] | 下一天每小时买电上限（为正数） |
    | SellLimitPerHour  |  Y      | float[] | 下一天每小时卖电上限（为正数） |
    | BatteryCostPerPower |  Y      | float   | 电池单位电量的成本 |
    | SOCMax            |  Y      | float   | 电池可用容量上限 |
    | SOCMin            |  Y      | float   | 电池可用容量下限 |
    | BatteryUsageMax   |  Y      | float   | 电池每小时放电量上限（为正数） |
    | BatteryChargeMax  |  Y      | float   | 电池每小时充电量上限（为正数） |
    | PlanType          |  Y      | int     | 0为计算每小时买卖电量上限<br>1为不限制买卖电量上限（SLimit、BLimit无效） |
    
    ### 返回参数
    |    参数名  |  类型    |  说明 |
    |-----------|:-------:|-------|
    | code      |  int   | 状态码    |
    | success   |  int   | 是否成功  |
    | message   | string | 失败时的错误信息 |
    | data      | object | 返回数据 |
    | 1 Success         |  int    | 1为规划成功<br>0为规划失败，条件冲突 |
    | 2 ShortagePerHour | float[] | 每小时用电量缺口，为用户用电量减去光伏发电量（负代表可用于卖电） |
    | 3 PlanBuyPerHour  | float[] | 规划的每小时买电量（负为卖电量） |
    | 4 BatteryPlanPerHour | float[] | 规划的每小时电池放电量（正为放电，负为充电） |
    | 5 SOCPerHour      | float[] | 规划的每小时截止时的SOC剩余容量 |
    | 6 PlanCostPerHour | float[] | 规划的每小时的成本（负代表减去电池成本后的卖电收益） |
    | 7 TotalCost       | float   | 规划的总成本（成本为买电费用+电池使用成本） |
  
    ### 请求示例
    ```json
    {
        "T":24,
        "BatteryPerHour":[50,50],
        "UsagePerHour":[100,100,100,120,130,120,110,100,120,140,160,170,110,100,90,80,70,60,50,40,30,20,10,10],
        "GeneratePerHour":[0,0,0,10,20,30,40,40,30,20,10,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "BuyPricePerHour":[0.57,0.77,0.77,0.77,0.57,0.57,0.57,0.57,0.77,0.77,0.77,0.77,1.03,1.03,1.03,0.57,0.26,0.26,0.26,0.26,0.26,0.26,0.26,0.26],
        "SellPricePerHour":[0.27,0.32,0.32,0.32,0.27,0.27,0.27,0.27,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.27,0.19,0.19,0.19,0.19,0.19,0.19,0.19,0.19],
        "SellLimitPerHour":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "BuyLimitPerHour":[800,800,800,200,200,200,200,200,200,120,120,120,800,800,800,800,800,800,800,800,800,800,800,800],
        "BatteryCostPerPower":0.1,
        "SOCMax":200,
        "SOCMin":20,
        "BatteryUsageMax":60,
        "BatteryChargeMax":50,
        "PlanType":0
    }
    ```

    ### 返回示例
    ```json
    {
        "data":{
            "Success":1,
            "SOCPerHour":[150,100,99.76,99.52,124.64,149.76,174.88,200,200,160,110,60,46.67,33.33,20,20,42.5,65,87.5,110,132.5,155,177.5,200],
            "BatteryPlanPerHour":[50,50,0.24,0.24,-25.12,-25.12,-25.12,-25.12,0,40,50,50,13.33,13.33,13.33,0,-22.5,-22.5,-22.5,-22.5,-22.5,-22.5,-22.5,-22.5],
            "ShortagePerHour":[100,100,100,130,150,150,150,140,150,160,170,170,110,100,90,80,70,60,50,40,30,20,10,10],
            "PlanBuyPerHour":[50,50,99.76,129.76,175.12,175.12,175.12,165.12,150,120,120,120,96.67,86.67,76.67,80,92.5,82.5,72.5,62.5,52.5,42.5,32.5,32.5],
            "PlanCostPerHour":[33.5,43.5,76.84,99.94,102.33,102.33,102.33,96.63,115.5,96.4,97.4,97.4,100.9,90.6,80.3,45.6,26.3,23.7,21.1,18.5,15.9,13.3,10.7,10.7],
            "TotalCost":1521.7
        },
        "success":1,
        "message":"",
        "code":200
    }
    ```
    """
    try:
        jdata = json.loads(request.get_data().decode("utf-8"))
        uph  = jdata.get('UsagePerHour')
        gph  = jdata.get('GeneratePerHour')
        bpph = jdata.get('BuyPricePerHour')
        spph = jdata.get('SellPricePerHour')
        blph = jdata.get('BuyLimitPerHour')
        slph = jdata.get('SellLimitPerHour')
        bcpp  = float(jdata.get('BatteryCostPerPower'))
        socmax = float(jdata.get('SOCMax'))
        socmin = float(jdata.get('SOCMin'))
        bum  = float(jdata.get('BatteryUsageMax'))
        bcm  = float(jdata.get('BatteryChargeMax'))
        ptype = int(jdata.get('PlanType'))
        bph = jdata.get('BatteryPerHour')
        T = jdata.get('T')
        if T is None:
            model = BatteryOptimizer(24)
        else:
            model = BatteryOptimizer(int(T))
        model.set_params(uph, gph, bpph, spph, blph, slph, bcpp, socmax, socmin, bum, bcm, ptype)
        ret = model.calc_plan_in_day(bph)
        return json.dumps({'data':ret,"success":1, "message":"","code":200}, ensure_ascii=False)
    except:
        msg = traceback.format_exc()
        LogManager.Instance.Log(__name__, 2, 'calc_plan error\n' + msg)
        return json.dumps({'data':msg,"success":0, "message":"查询失败","code":403}, ensure_ascii=False)

@blue.route('/battery/calcCost',methods=['post','get'])
def calc_cost():
    """成本计算
    ### 说明
    通过给出相关数据，计算每小时成本与总成本。该接口返回数据保留两位小数。

    ### 请求参数
    |      参数名         | 是否必填 |  类型    | 说明      |
    |--------------------|:-------:|:-------:|----------|
    | UsagePerHour       |  Y      | float[] | 每小时用电量（如果T为48则代表每半小时，数组长度应对应T，下同） |
    | GeneratePerHour    |  Y      | float[] | 每小时发电量（负为发电，正为用电）|
    | BuyPricePerHour    |  Y      | float[] | 每小时买电电价 |
    | SellPricePerHour   |  Y      | float[] | 每小时卖电电价 |
    | BatteryCostPerPower |  Y      | float   | 电池单位电量的成本 |
    | BatteryPerHour     |  Y      | float[] | 电池每小时放电量（正为放电，负为充电） |

    ### 返回参数
    |    参数名  |  类型    |  说明 |
    |-----------|:-------:|-------|
    | code      |  int   | 状态码    |
    | success   |  int   | 是否成功  |
    | message   | string | 失败时的错误信息 |
    | data      | object | 返回数据 |
    | 1 CostPerHour | float[] | 规划的每小时的成本（负代表减去电池成本后的卖电收益） |
    | 2 TotalCost   | float   | 规划的总成本（成本为买电费用+电池使用成本） |
  
    ### 请求示例
    ```json
    {
        "UsagePerHour":[100,100,100,120,130,120,110,100,120,140,160,170,110,100,90,80,70,60,50,40,30,20,10,10],
        "GeneratePerHour":[0,0,0,10,20,30,40,40,30,20,10,0,0,0,0,0,0,0,0,0,0,0,0,0],
        "BuyPricePerHour":[0.57,0.77,0.77,0.77,0.57,0.57,0.57,0.57,0.77,0.77,0.77,0.77,1.03,1.03,1.03,0.57,0.26,0.26,0.26,0.26,0.26,0.26,0.26,0.26],
        "SellPricePerHour":[0.27,0.32,0.32,0.32,0.27,0.27,0.27,0.27,0.32,0.32,0.32,0.32,0.32,0.32,0.32,0.27,0.19,0.19,0.19,0.19,0.19,0.19,0.19,0.19],
        "BatteryCostPerPower":0.1,
        "BatteryPerHour":[0.24,2.6,2.6,2.6,-2.01,-2.01,-2.01,-2.01,0,40,50,50,13.33,13.33,13.33,-2.03,-22.25,-22.25,-22.25,-22.25,-22.25,-22.25,-22.25,-22.25]   
    }
    ```

    ### 返回示例
    ```json
    {"data":{"TotalCost":1512.41,"CostPerHour":[56.89,75.26,75.26,98.36,86.85,86.85,86.85,81.15,115.5,96.4,97.4,97.4,100.9,90.6,80.3,46.96,26.21,23.61,21.01,18.41,15.81,13.21,10.61,10.61]},"success":1,"message":"","code":200}
    ```
    """
    try:
        jdata = json.loads(request.get_data().decode("utf-8"))
        uph  = jdata.get('UsagePerHour')
        gph  = jdata.get('GeneratePerHour')
        bpph  = jdata.get('BuyPricePerHour')
        spph  = jdata.get('SellPricePerHour')
        bcpp  = float(jdata.get('BatteryCostPerPower'))
        bph = jdata.get('BatteryPerHour')
        cost = 0
        costbyday = []
        for i in range(len(uph)):
            dcost = abs(bph[i]) * bcpp
            gp = uph[i] + gph[i] - bph[i]
            if gp > 0:  #买电
                dcost += gp * bpph[i]
            else:       #卖电
                dcost += gp * spph[i]
            dcost = round(dcost, 2)
            cost += dcost
            costbyday.append(dcost)
        ret = {'TotalCost': round(cost,2), 'CostPerHour':costbyday}
        return json.dumps({'data':ret,"success":1, "message":"","code":200}, ensure_ascii=False)
    except:
        msg = traceback.format_exc()
        LogManager.Instance.Log(__name__, 2, 'calc_cost error\n' + msg)
        return json.dumps({'data':msg,"success":0, "message":"查询失败","code":403}, ensure_ascii=False)