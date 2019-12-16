from ParserConf import ParserConf

app_conf = ParserConf('dualpc.ini')
app_conf.parserConf()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from dualpc import dualpc
model = dualpc(app_conf)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.setDevice(device)

from DataUtil import DataUtil
data = DataUtil(app_conf)

import train as train
train.start(app_conf, data, model)