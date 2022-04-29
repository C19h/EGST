import os,io,base64,json, sys, traceback
from flask import Blueprint, render_template, redirect, send_from_directory, request, Response
from logmanager import LogManager
from configmanager import ConfigManager

blue = Blueprint('info',__name__)

@blue.route('/api/getLogData',methods=['post'])
def getLogData():
    jdata = json.loads(request.get_data().decode("utf-8"))
    cinfo  = int(jdata.get('info'))
    cwarning  = int(jdata.get('warning'))
    cerror  = int(jdata.get('error'))
    cdebug  = int(jdata.get('debug'))
    return json.dumps(LogManager.Instance.GetLog(cinfo, cwarning, cerror, cdebug))

@blue.route('/',methods=['get'])
def index():
    return send_from_directory("./web","index.html")

@blue.route('/html/<file>',methods=['get'])
def html(file):
    return send_from_directory("./web/html",file)

@blue.route('/resource/<p1>/<p2>/<file>')
def resource3(p1,p2,file):
    return send_from_directory("./web/resource/%s/%s"%(p1,p2),file)

@blue.route('/resource/<p1>/<file>')
def resource2(p1,file):
    return send_from_directory("./web/resource/%s"%p1,file)

@blue.route('/resource/<path>')
def resource1(path):
    return send_from_directory("./web/resource",path)