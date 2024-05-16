import torch
from torch.nn import functional as F
from models.RLFN.rlfn_s import RLFN_S


class RLFN:
    def __init__(self, sr_factor=2) -> None:
        self.sr_factor = sr_factor
        self.net = RLFN_S(in_channels=3, out_channels=3, upscale=self.sr_factor)

    def eval(self):
        self.net.eval()

    def device(self):
        self.net.to(torch.device("cuda"))
    
    def load_model(self):
        self.net.load_state_dict(torch.load((f'./ckpt/RLFN/rlfn_s_x{int(self.sr_factor)}.pth')), strict=True)
        for k, v in self.net.named_parameters():
            v.requires_grad = False
    
    @torch.no_grad()
    def inference(self, img_L):
        img_L *= 255.
        img_H = self.net(img_L).clip(0., 255.)
        img_H /= 255.
        return img_H