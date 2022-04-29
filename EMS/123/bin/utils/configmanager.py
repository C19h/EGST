import configparser
import base64


class ConfigManager(object):
    def __init__(self, filename):
        self.gConfig = configparser.ConfigParser()
        self.gFilename = filename
        ConfigManager.Instance = self

    def GetConfig(self, section, key):
        self.__read()
        return self.gConfig.get(section, key.lower())

    def GetConfigDefaultBase64(self, section, key, default):
        try:
            ret = base64.b64decode(self.GetConfig(section, key).encode("utf8")).decode('utf8')
        except:
            ret = default
            self.SetConfigBase64(section, key, ret)
        return ret

    def GetConfigDefault(self, section, key, default):
        try:
            ret = self.GetConfig(section, key)
        except:
            ret = default
            self.SetConfig(section, key, ret)
        return ret

    def SetConfig(self, section, key, val):
        self.__read()
        try:
            self.gConfig.add_section(section)
        except:
            pass
        self.gConfig.set(section, key.lower(), val)
        self.__write()

    def SetConfigBase64(self, section, key, val):
        self.__read()
        try:
            self.gConfig.add_section(section)
        except:
            pass

        self.gConfig.set(section, key.lower(), base64.b64encode(val.encode('utf8')).decode('utf8'))
        self.__write()

    def __read(self):
        self.gConfig.read([self.gFilename], encoding='utf8')  # 读取配置文件，如果写文件的绝对路径，就可以不用os模块

    def __write(self):
        with open(self.gFilename, 'w', encoding='utf8') as f:
            self.gConfig.write(f)


ConfigManager('config.ini')
