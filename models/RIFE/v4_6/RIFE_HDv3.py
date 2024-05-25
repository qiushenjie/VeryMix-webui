import torch
import torch.nn as nn
from models.RIFE.warplayer import warp
from torch.nn.parallel import DistributedDataParallel as DDP
from models.RIFE.v4_6.IFNet_HDv3 import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Model:
    def __init__(self, local_rank=-1):
        self.flownet = IFNet()
        self.device()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path))), False)
            else:
                self.flownet.load_state_dict(convert(torch.load('{}/flownet.pkl'.format(path), map_location ='cpu')), False)
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    @torch.no_grad()
    def inference(self, img0, img1, timestep=0.5, scale_factor=1.0, TTA=False, fast_TTA=False):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [8/scale_factor, 4/scale_factor, 2/scale_factor, 1/scale_factor]
        _, _, pred = self.flownet(imgs, timestep, scale_list)
        pred = pred[3]
        '''
        Infer with scale_factor=1.0 flow
        Noting: return BxCxHxW
        '''
        TTA = fast_TTA
        if TTA == False:
            return pred
        else:
            _, _, pred2 = self.flownet(imgs.flip(2).flip(3), timestep, scale_list)
            pred2 = pred2[3]
            return (pred + pred2.flip(2).flip(3)) / 2

    @torch.no_grad()
    def multi_inference(self, img0, img1, time_list=[], scale_factor=1.0, TTA=False, fast_TTA=False):
        '''
        Run backbone once, get multi frames at different timesteps
        Noting: return a list of [CxHxW]
        '''
        assert len(time_list) > 0, 'Time_list should not be empty!'

        TTA = fast_TTA
        def infer(imgs):
            pred_list = []
            for timestep in time_list:
                pred = self.forward(imgs, timestep, scale_factor)
                pred_list.append(pred)

            return pred_list

        imgs = torch.cat((img0, img1), 1)

        preds = infer(imgs)
        if TTA is False:
            return [preds[i][0] for i in range(len(time_list))]
        else:
            flip_pred = infer(imgs.flip(2).flip(3))
            return [(preds[i][0] + flip_pred[i][0].flip(1).flip(2))/2 for i in range(len(time_list))]

