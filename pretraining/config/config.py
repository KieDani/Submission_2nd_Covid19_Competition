import sys
import torch
import os
import socket
from socket import gethostname
from pretraining.config.modelconfig import get_modelconfig

from pretraining.config.dataconfig import get_cvsplit_patients, get_dataconfig
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime as datetimefunction

from paths import pretraining_LOGS_PATH, pretraining_PRETRAINED_PATH, pretraining_DATA_PATH

#from config.dataconfig import STOICDataConfig, STOICCachedDataConfig


def get_ip_end():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    # print(int(IP[IP.rfind(".") + 1:]))
    return int(IP[IP.rfind(".") + 1:])



class AllConfig:
    """
    Base class for configuration files of all kinds
    """

    def __init__(self, args):
        #self.GPUS = [6]
        self.GPUS = args.gpu

        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu) for gpu in self.GPUS])

        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        #self.DEVICE = 'cpu'
        self.DEVICE = torch.device(self.DEVICE)

        if sys.gettrace() is None:
            self.WORKERS = 8
        else:
            self.WORKERS = 0

        self.IMAGE_SIZE = args.imsize

        self.DATA_PATH = pretraining_DATA_PATH
        self.LOGS_PATH = pretraining_LOGS_PATH
        self.PRETRAINED_PATH = pretraining_PRETRAINED_PATH

        #TODO create PATH for file to save the configuration of the run

        self.Batch_SIZE = 4
        self.MAX_EPOCHS = 400
        self.REDUCTION_FACTOR = 15

        self.MODEL_NAME = args.model
        #self.UNSUPERVISED_DATA = 'tcia'
        self.UNSUPERVISED_DATA = 'stoic'
        #self.UNSUPERVISED_DATA = 'mosmed'
        self.LOSS = 'dicece' #'balancedce'
        self.DO_NORMALIZATION = True
        self.SUPERVISED_AUGMENTATIONS = False

        self.modelconfig = get_modelconfig(self)