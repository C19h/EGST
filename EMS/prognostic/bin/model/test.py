# author:c19h
# datetime:2022/4/26 14:37
import requests, json
from bin.utils.dataloader import DataLoader

rooturl = 'http://127.0.0.1:9876'


def postrequest(req, body):
    try:
        url = rooturl + req
        headers = {'content-type': "application/json", 'Authorization': 'APP appid = 4abf1a,token = 9480295ab2e2eddb8'}
        datastr = json.dumps(body)
        response = requests.post(url, data=datastr, headers=headers)
        ret = json.loads(response.text)
        print(ret)
        return ret
    except Exception as e:
        print(e)
        return None


def main():
    ld = DataLoader('./bin/utils/config.ini')
    ld.load_data('first_cluster_cell_voltage', 0, 3)
    datas = ld.datas
    json_values = datas.to_json(orient="index")
    data = {'region': 'inner', 'data': json_values, 'window': 10, 'amplitude': 0.01, 'time_step': 30, 'threshold': 0.95,
            'tolerance': 5}
    postrequest('/security', data)


if __name__ == "__main__":
    main()
