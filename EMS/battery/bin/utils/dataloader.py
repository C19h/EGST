import csv, datetime

class DataLoader(object):
    def __init__(self):
        self.Path_UPh = './data/load.csv'
        self.Path_LPh = './data/pv.csv'
        self.Data = {}
        
    def LoadData(self):
        lph = self.LoadLPh()
        uph = self.LoadUPh()
        data = {}
        for rk in lph:
            if not rk in uph:
                continue
            tk = rk[:-3]
            if not tk in data:
                data[tk] = []
            tmp = {'Time':rk, 'LPh': lph[rk], 'UPh':uph[rk]}
            tim = datetime.datetime.strptime(rk, "%Y-%m-%d %H")
            tim = tim + datetime.timedelta(hours=7)
            tmp['TrueTime'] = tim.strftime('%Y-%m-%d %H')
            data[tk].append(tmp)
            
        for rk in data:
            if len(data[rk]) != 24:
                continue
            self.Data[rk] = data[rk]
        return data

    def LoadUPh(self):
        data = {}
        with open(self.Path_UPh, 'r') as f:
            fcsv = csv.reader(f)
            for row in fcsv:
                tim = datetime.datetime.strptime(row[1], "%Y/%m/%d %H:%M")
                tim = tim - datetime.timedelta(hours=7)
                timstr = tim.strftime('%Y-%m-%d %H')
                if not timstr in data:
                    data[timstr] = []
                data[timstr].append(float(row[2]))
        for rowk in data:
            data[rowk] = sum(data[rowk]) / len(data[rowk])
        return data
    
    def LoadLPh(self):
        data = {}
        with open(self.Path_LPh, 'r') as f:
            fcsv = csv.reader(f)
            for row in fcsv:
                tim = datetime.datetime.strptime(row[1], "%m/%d/%Y %H:%M:%S")
                tim = tim - datetime.timedelta(hours=7)
                timstr = tim.strftime('%Y-%m-%d %H')
                if not timstr in data:
                    data[timstr] = []
                data[timstr].append(float(row[2]))
        for rowk in data:
            data[rowk] = sum(data[rowk]) / len(data[rowk])
        return data

if __name__ == "__main__":
    dl = DataLoader()
    dl.LoadData()