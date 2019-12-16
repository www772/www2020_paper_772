import configparser as cp
import re, os

class ParserConf():

    def __init__(self, config_path):
        self.config_path = config_path

    def processValue(self, key, value):
        #print(key, value)
        value = value.replace(',', '')
        tmp = value.split(' ')
        dtype = tmp[0]
        value = tmp[1:]
        #print(dtype, value)

        if 'show_list' in key:
            for i in value:
                self.show_list.append(i)

        if value != None:
            if dtype == 'string':
                self.conf_dict[key] = vars(self)[key] = value[0]
            elif dtype == 'int':
                self.conf_dict[key] = vars(self)[key] = int(value[0])
            elif dtype == 'float':
                self.conf_dict[key] = vars(self)[key] = float(value[0])
            elif dtype == 'list':
                self.conf_dict[key] = vars(self)[key] = [i for i in value]
            elif dtype == 'int_list':
                self.conf_dict[key] = vars(self)[key] = [int(i) for i in value]
            elif dtype == 'float_list':
                self.conf_dict[key] = vars(self)[key] = [float(i) for i in value]
        else:
            print('%s value is None' % key)

    def parserConf(self):
        conf = cp.ConfigParser()
        conf.read(self.config_path)
        self.conf = conf

        self.conf_dict = {}
        # sometimes the show_list may contain too many keys, we have to specify the show list with multilines, thus we 
        # store the show keys with a set
        self.show_list = []
        for section in conf.sections():
            for (key, value) in conf.items(section):
                #print(key, value)
                self.processValue(key, value)