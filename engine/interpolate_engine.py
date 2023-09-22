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
from models.EMAVFI.EMAVFI import EMAVFI  # pylint: disable=import-error
from models.RIFE.RIFE_HDv3 import Model as RIFE


class InterpolateEngine:
    def __init__(self, gpu_ids):
        """Iniitalize the class by calling into VFI code"""
        gpu_id_array = self.init_device(gpu_ids)
        self.model= None

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
            if model_name == 'EMAVFI Small':
                vfi_model = EMAVFI(name="emavfi_s", local_rank=-1)
                vfi_model.load_model(name="emavfi_s")
                print("load EMAVFI Small succeed!\n")
                vfi_model.eval()
                vfi_model.device()
                
            elif model_name == 'EMAVFI':
                vfi_model = EMAVFI(name="emavfi", local_rank=-1)
                vfi_model.load_model(name="emavfi")
                print("load EMAVFI succeed!\n")
                vfi_model.eval()
                vfi_model.device()
                
            elif model_name == 'RIFE v4.6':
                vfi_model = RIFE()
                vfi_model.load_model(os.path.join(__dir__, "../ckpt/RIFEv4.6"), -1)
                print("load RIFE v4.6 succeed!\n")
                vfi_model.eval()
                vfi_model.device()

            else:
                vfi_model = RIFE()
                vfi_model.load_model(os.path.join(__dir__, "../ckpt/RIFEv4.6"), -1)
                print("load RIFE v4.6 succeed!\n")
                vfi_model.eval()
                vfi_model.device()
                
            self.model = vfi_model
        except:
            vfi_model = RIFE()
            vfi_model.load_model(os.path.join(__dir__, "../ckpt/RIFEv4.6"), -1)
            print("load RIFE v4.6 succeed!\n")
            vfi_model.eval()
            vfi_model.device()
            self.model = vfi_model

    def release(self):
        del self.model
        torch.cuda.empty_cache()
        self.model = None
