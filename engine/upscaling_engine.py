from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import sys
import torch

'''==========import from our code=========='''
from models.RLFN.RLFN import RLFN
from models.RealESRGAN.RealESRGAN import RealESRGAN
from models.ShuffleCUGAN.ShuffleCUGAN import ShuffleCUGAN


class UpscalingEngine:
    def __init__(self, gpu_ids, sr_factor=2):
        """Iniitalize the class by calling into sr code"""
        gpu_id_array = self.init_device(gpu_ids)
        self.model= None
        self.sr_factor = sr_factor

    def init_device(self, gpu_ids : str):
        """for *future use*"""
        str_ids = gpu_ids.split(',')
        gpu_ids = []
        for str_id in str_ids:
            _id = int(str_id)
            if _id >= 0:
                gpu_ids.append(_id)
        # for *future use*
        # if len(gpu_ids) > 0:
        #     torch.cuda.set_device(gpu_ids[0])
        # cudnn.benchmark = True
        return gpu_ids
    
    def load(self, model_name):
        try:
            if model_name == 'RLFN':
                sr_model = RLFN(sr_factor=self.sr_factor)
                sr_model.load_model()
                print("load RLFN succeed!\n")
                sr_model.eval()
                sr_model.device()

            elif model_name == 'RealESRGAN':
                sr_model = RealESRGAN(sr_factor=self.sr_factor)
                sr_model.load_model()
                print("load RealESRGAN succeed!\n")
                sr_model.eval()
                sr_model.device()

            elif model_name == 'ShuffleCUGAN':
                sr_model = ShuffleCUGAN(sr_factor=self.sr_factor)
                sr_model.load_model()
                print("load ShuffleCUGAN succeed!\n")
                sr_model.eval()
                sr_model.device()

            else:
                sr_model = RealESRGAN(sr_factor=self.sr_factor)
                sr_model.load_model()
                print("load RealESRGAN succeed!\n")
                sr_model.eval()
                sr_model.device()
                
            self.model = sr_model
        except:
            sr_model = RLFN(sr_factor=self.sr_factor)
            sr_model.load_model()
            print("load RLFN succeed!\n")
            sr_model.eval()
            sr_model.device()
            self.model = sr_model

    def release(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = None
