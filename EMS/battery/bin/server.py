#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Author:huxiao
# 网站服务页
# ==============================================================================
from flask import Flask, Response, send_file, request, render_template, send_from_directory
import os, io, base64, json, sys, traceback

sys.path.append("./bin")
sys.path.append("./bin/utils")
sys.path.append("./bin/webservice")
sys.path.append("./bin/work")
from flask_docs import ApiDoc
from logmanager import LogManager
from configmanager import ConfigManager

server = Flask(__name__)

API_DOC_ROUTER_NAME = [
    ['battery', '峰谷套利优化计算接口']
]

checkreturn = ConfigManager.Instance.GetConfigDefault("Server", "CheckReturnParam", "0")


def loadConfig():
    port = ConfigManager.Instance.GetConfigDefault("Server", "ListenPort", "9998")
    port = int(port)
    LogManager.Instance.Log(__name__, 0, "Server start.(port=%d)" % port)
    print('======server======', port)
    return port


###### check param
type2instance = {
    'string': str,
    'int': float,
    'float': float,
    'object': dict,
    'list': list,
    'dict': dict,
    'bool': bool
}

rules = {}
rules_after = {}


def _parse_doc(doc):
    if not doc is None:
        paras = doc.split('### ')
        for para in paras:
            if para[:4] == '请求参数':
                rows = para.split('\n')[3:]
                obj = {'layer': 0, 'data': []}
                _make_obj(rows, obj)
                return obj
    return None


def _parse_doc_after(doc):
    if not doc is None:
        paras = doc.split('### ')
        for para in paras:
            if para[:4] == '返回参数':
                rows = para.split('\n')[3:]
                obj = {'layer': 0, 'data': []}
                _make_obj_after(rows, obj)
                return obj
    return None


def _make_type(ptype, data):
    if ptype.find('-') > 0:
        tmp = {'data': [], 'layer': data['layer']}
        ttype = ptype.split('-', 1)
        ret = _make_type(ttype[1], tmp)
        data['type'] = dict
        data['type-dict'] = type2instance[ttype[0]]
        data['ntype'] = ptype
        data['child'] = tmp
        return ret
    elif ptype[-2:] == '[]':
        tmp = {'data': [], 'layer': data['layer']}
        ret = _make_type(ptype[:-2], tmp)
        data['type'] = list
        data['ntype'] = ptype
        data['child'] = tmp
        return ret
    else:
        data['type'] = type2instance[ptype]
        data['ntype'] = ptype
        return data


def _make_obj(rows, obj):
    if not 'layer' in obj:
        return
    rootlayer = obj['layer']
    while (len(rows) > 0):
        row = rows[0]
        cols = row.split('|')
        if len(cols) < 2:
            break
        name = cols[1].strip()
        layer, name = _get_layer(name)
        if layer == rootlayer:
            # 同层需处理
            tmp = {'name': name, 'data': [], 'layer': rootlayer}
            need = cols[2].strip().lower()
            if need == 'y':
                tmp['need'] = True
            else:
                tmp['need'] = False
            ptype = cols[3].strip().lower()
            child = _make_type(ptype, tmp)
            obj['data'].append(tmp)
            rows.pop(0)
            child['layer'] += 1
            _make_obj(rows, child)
        elif layer > rootlayer:
            # 下一层
            child = {'layer': layer, 'data': []}
            tmp['child'] = child
            _make_obj(rows, child)
        else:
            # 上一层，直接退出
            return


def _make_obj_after(rows, obj):
    if not 'layer' in obj:
        return
    rootlayer = obj['layer']
    while (len(rows) > 0):
        row = rows[0]
        cols = row.split('|')
        if len(cols) < 2:
            break
        name = cols[1].strip()
        layer, name = _get_layer(name)
        if layer == rootlayer:
            # 同层需处理
            tmp = {'name': name, 'data': [], 'layer': rootlayer}
            ptype = cols[2].strip().lower()
            if ptype[0] == '*':
                tmp['need'] = False
                ptype = ptype[1:-1]
            else:
                tmp['need'] = True
            child = _make_type(ptype, tmp)
            obj['data'].append(tmp)
            rows.pop(0)
            child['layer'] += 1
            _make_obj_after(rows, child)
        elif layer > rootlayer:
            # 下一层
            child = {'layer': layer, 'data': []}
            tmp['child'] = child
            _make_obj_after(rows, child)
        else:
            # 上一层，直接退出
            return


def _get_layer(name):
    sp1 = name.split(' ')
    if len(sp1) == 1:
        return 0, name
    else:
        sp2 = sp1[0].split('.')
        return len(sp2), sp1[-1]


def _check_data_type(clstype, data):
    if clstype == float and isinstance(data, int):  # int也算float
        return True
    elif isinstance(data, clstype):
        return True
    return False


def _check_type(par, data):
    if data is None:  # None也算通过了检查
        return None
    if _check_data_type(par['type'], data) == True:
        if par['ntype'] == 'object':
            return _check_data(par, data)
        elif par['type'] == list:
            for dat in data:
                if dat is None:
                    continue
                ret = _check_type(par['child'], dat)
                if ret != None:
                    return ret
            return None
        elif par['type'] == dict and 'child' in par:
            for strkey in data.keys():
                if _check_data_type(par['type'], strkey) == False:
                    return par['name'], "key类型错误"
                dat = data[strkey]
                if dat is None:
                    continue
                ret = _check_type(par['child'], dat)
                if ret != None:
                    return ret
        else:
            return None
    else:  # 参数类型错误
        return par['name'], "类型错误"


def _check_data(param, data):
    for par in param['data']:
        if par['name'] in data:  # 该参数存在，进行检查
            ret = _check_type(par, data[par['name']])
            if ret != None:
                return ret
        elif par['need'] == True:  # 该参数不存在，但需要
            return par['name'], "不存在"
        else:
            continue


@server.before_request
def bef(*args, **kwargs):
    global rules
    if request.path in rules:
        if request.content_type.lower().find('json') < 0:
            return json.dumps(
                {"success": 0, "message": "请求头中CONTENT_TYPE需要application/json", "code": 417, "data": None},
                ensure_ascii=False), 200, {"Content-Type": "application/json"}
        try:
            jdata = json.loads(request.get_data().decode("utf-8"))
        except:
            return json.dumps({"success": 0, "message": "数据不是合法的json格式", "code": 417, "data": None},
                              ensure_ascii=False), 200, {"Content-Type": "application/json"}
        ret = _check_data(rules[request.path], jdata)
        if ret is None:
            return
        else:
            LogManager.Instance.Log(__name__, 2, request.path + "\n参数校验未通过，%s%s" % ret)
            return json.dumps({"success": 0, "message": "参数校验未通过，%s%s" % ret, "code": 417, "data": None},
                              ensure_ascii=False), 200, {"Content-Type": "application/json"}
    else:
        return


@server.after_request
def aef(response):
    global rules_after
    if response.status_code != 200:
        return response
    if request.path in rules_after:
        response.content_type = 'application/json'
        if checkreturn == "0":
            return response
        jdata = json.loads(response.get_data().decode("utf-8"))
        if jdata['success'] != 1:
            return response
        ret = _check_data(rules_after[request.path], jdata)
        if ret is None:
            return response
        else:
            LogManager.Instance.Log(__name__, 2, request.path + "\n返回参数校验未通过，%s%s" % ret)
            response.set_data(json.dumps({"success": 0, "message": "返回参数校验未通过，%s%s" % ret, "code": 417, "data": jdata},
                                         ensure_ascii=False))
            return response
    else:
        return response


def generate_rules(app):
    global rules
    for rule in app.url_map.iter_rules():
        func = app.view_functions[rule.endpoint]
        doc = func.__doc__
        data = _parse_doc(doc)
        if not data is None and len(data['data']) > 0:
            rules[rule.rule] = data
        data_after = _parse_doc_after(doc)
        if not data_after is None and len(data_after['data']) > 0:
            rules_after[rule.rule] = data_after


######

def runServer(port):
    # 采用动态加载的方式
    server.config["API_DOC_METHODS_LIST"] = ["POST"]
    server.config.setdefault("API_DOC_ROUTER_NAME", API_DOC_ROUTER_NAME)
    ApiDoc(
        server,
        title="电池峰谷套利优化计算系统",
        version="1.0.0",
        description="FFP",
    )
    for fname in os.listdir('./bin/webservice'):
        if fname[-10:] != 'handler.py':
            continue
        try:
            module = __import__(fname[:-3])
            blue = getattr(module, 'blue')
            server.register_blueprint(blue, url_prefix='/')
        except:
            msg = traceback.format_exc()
            data = '模块加载错误: %s\n%s' % (fname, msg)
            print(data)
            LogManager.Instance.Log(__name__, 2, data)

    generate_rules(server)
    server.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    port = loadConfig()
    runServer(port)
