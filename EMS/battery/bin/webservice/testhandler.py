#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:huxiao
# 网站服务页
# ==============================================================================
import os,io,base64,json, sys, traceback, random
from flask import Blueprint, render_template, redirect, send_from_directory, request, Response
import pickle
from logmanager import LogManager
from dataloader import DataLoader
from batteryoptimizer import BatteryOptimizer

blue = Blueprint('test',__name__)

gData = DataLoader()
gData.LoadData()

@blue.route('/test/getDays',methods=['post','get'])
def get_days():
    global gData
    ret = []
    for rk in gData.Data:
        ret.append(rk)
    return json.dumps(ret)

@blue.route('/test/getDataByDay',methods=['post'])
def get_data_by_day():
    try:
        global gData
        jdata = json.loads(request.get_data().decode("utf-8"))
        datestr = jdata.get('day')
        UPh = []
        LPh = []
        for row in gData.Data[datestr]:
            UPh.append(round(row['UPh'],2))
            LPh.append(round(row['LPh']* -1,2))
        return json.dumps({'data':{'GeneratePerHour':LPh, 'UsagePerHour':UPh},"success":1, "message":"","code":200}, ensure_ascii=False)
    except:
        msg = traceback.format_exc()
        LogManager.Instance.Log(__name__, 2, 'get_data_by_day error\n' + msg)
        return json.dumps({'data':msg,"success":0, "message":"查询失败","code":403}, ensure_ascii=False)


@blue.route('/getChart3',methods=['post','get'])
def getChart3():
    global gData
    jdata = json.loads(request.get_data().decode("utf-8"))
    datestr = jdata.get('datestr')
    UPh = []
    LPh = []
    for row in gData.Data[datestr]:
        UPh.append(row['UPh'])
        LPh.append(row['LPh']* -1)
    BCost  = float(jdata.get('BCost'))
    SPh  = json.loads(jdata.get('SPh'))
    BPh  = json.loads(jdata.get('BPh'))
    SLimit  = json.loads(jdata.get('SLimit'))
    BLimit  = json.loads(jdata.get('BLimit'))
    SOCMax  = float(jdata.get('SOCMax'))
    SOCMin  = float(jdata.get('SOCMin'))
    BUMax  = float(jdata.get('BUMax'))
    BCMax  = float(jdata.get('BCMax'))
    model = BatteryOptimizer(24)
    model.set_params(UPh, LPh, BPh, SPh, BLimit, SLimit, BCost, SOCMax, SOCMin, BUMax, BCMax)
    ret = model.calc_plan()
    for i in range(24):
        #第一个点波动20%
        UPh[i] = UPh[i] * (0.8 + random.random()* 0.4)
        model = BatteryOptimizer(24)
        model.set_params(UPh, LPh, BPh, SPh, BLimit, SLimit, BCost, SOCMax, SOCMin, BUMax, BCMax)
        ret = model.calc_plan_in_day(ret['U'][:i])
    return json.dumps(ret)

@blue.route('/getChart2',methods=['post','get'])
def getChart2():
    global gData
    jdata = json.loads(request.get_data().decode("utf-8"))
    datestr = jdata.get('datestr')
    UPh = []
    LPh = []
    for row in gData.Data[datestr]:
        UPh.append(row['UPh'])
        LPh.append(row['LPh']* -1)
    BCost  = float(jdata.get('BCost'))
    SPh  = json.loads(jdata.get('SPh'))
    BPh  = json.loads(jdata.get('BPh'))
    SLimit  = json.loads(jdata.get('SLimit'))
    BLimit  = json.loads(jdata.get('BLimit'))
    SOCMax  = float(jdata.get('SOCMax'))
    SOCMin  = float(jdata.get('SOCMin'))
    BUMax  = float(jdata.get('BUMax'))
    BCMax  = float(jdata.get('BCMax'))
    Type = int(jdata.get('Type'))
    model = BatteryOptimizer(24)
    model.set_params(UPh, LPh, BPh, SPh, BLimit, SLimit, BCost, SOCMax, SOCMin, BUMax, BCMax, Type)
    ret = model.calc_plan()
    return json.dumps(ret)

@blue.route('/api/getChart2InDay',methods=['post','get'])
def getChart2InDay():
    jdata = json.loads(request.get_data().decode("utf-8"))
    datestr = jdata.get('datestr')
    UPh = []
    LPh = []
    for row in gData.Data[datestr]:
        UPh.append(row['UPh'])
        LPh.append(row['LPh']* -1)
    BCost  = float(jdata.get('BCost'))
    SPh  = json.loads(jdata.get('SPh'))
    BPh  = json.loads(jdata.get('BPh'))
    SLimit  = json.loads(jdata.get('SLimit'))
    BLimit  = json.loads(jdata.get('BLimit'))
    SOCMax  = float(jdata.get('SOCMax'))
    SOCMin  = float(jdata.get('SOCMin'))
    BUMax  = float(jdata.get('BUMax'))
    BCMax  = float(jdata.get('BCMax'))
    Hour = int(jdata.get('Hour'))
    Type = int(jdata.get('Type'))
    Battery = json.loads(jdata.get('Battery'))
    model = BatteryOptimizer(24)
    model.set_params(UPh, LPh, BPh, SPh, BLimit, SLimit, BCost, SOCMax, SOCMin, BUMax, BCMax, Type)
    ret = model.calc_plan_in_day(Battery)
    return json.dumps(ret)

@blue.route('/getChart',methods=['post','get'])
def getChart():
    jdata = json.loads(request.get_data().decode("utf-8"))
    BCost  = float(jdata.get('BCost'))
    UPh  = json.loads(jdata.get('UPh'))
    LPh  = json.loads(jdata.get('LPh'))
    SPh  = json.loads(jdata.get('SPh'))
    BPh  = json.loads(jdata.get('BPh'))
    SLimit  = json.loads(jdata.get('SLimit'))
    BLimit  = json.loads(jdata.get('BLimit'))
    SOCMax  = float(jdata.get('SOCMax'))
    SOCMin  = float(jdata.get('SOCMin'))
    BUMax  = float(jdata.get('BUMax'))
    BCMax  = float(jdata.get('BCMax'))
    model = BatteryOptimizer(24)
    model.set_params(UPh, LPh, BPh, SPh, BLimit, SLimit, BCost, SOCMax, SOCMin, BUMax, BCMax)
    ret = model.calc_plan()
    return json.dumps(ret)
