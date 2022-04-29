import time

class LogManager(object):
    def __init__(self, maxnum = 100):
        self.gMaxNum = maxnum
        self.gLog = []
        LogManager.Instance = self
        pass

    def Log(self, module, level, msg):
        print(msg)
        tobj = {}
        tobj['Time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        tobj['Message'] = msg
        if level == 0:
            tobj['Level'] = 'Info'
        elif level == 1:
            tobj['Level'] = 'Warning'
        elif level == 2:
            tobj['Level'] = 'Error'
        elif level == 3:
            tobj['Level'] = 'Debug'  
        self.gLog.insert(0, tobj)
        while (len(self.gLog) > self.gMaxNum):
            self.gLog.pop()

    def GetLog(self, info, warning, error, debug):
        ret = []
        for log in self.gLog:
            if (info > 0 and log['Level'] == 'Info'):
                ret.append(log)
            elif (warning > 0 and log['Level'] == 'Warning'):
                ret.append(log)
            elif (error > 0 and log['Level'] == 'Error'):
                ret.append(log)
            elif (debug > 0 and log['Level'] == 'Debug'):
                ret.append(log)
        return ret

LogManager(200)