# author:c19h
# datetime:2022/4/26 11:25
from flask import Flask, Response, send_file, request, render_template, send_from_directory
import json
from bin.utils.security_warning import SecurityWarning
import numpy as np
import pandas as pd
import scipy.stats
from pro import Prognostic

server = Flask(__name__)
server.debug = True


@server.route('/security', methods=['post', 'get'])
def security():
    try:
        jdata = json.loads(request.get_data().decode("utf-8"))
        region = jdata.get('region')
        data = jdata.get('data')
        window = jdata.get('window')
        amplitude = jdata.get('amplitude')
        time_step = jdata.get('time_step')
        threshold = jdata.get('threshold')
        tolerance = jdata.get('tolerance')
        p = Prognostic(region, data, window, amplitude, time_step, threshold, tolerance)
        relativity, num_abnormal, index = p.execute()
        data = {'fault': '正常', '相关性系数': relativity, '不正常数量': num_abnormal, '下标': index}
        return json.dumps({'data': data, "success": 1, "message": "", "code": 0}, ensure_ascii=False)
    except:
        return json.dumps({'data': None, "success": 0, "message": "失败", "code": 403}, ensure_ascii=False)


server.run(host="0.0.0.0", port=9876)
