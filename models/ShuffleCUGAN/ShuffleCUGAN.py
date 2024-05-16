import torch


class ShuffleCUGAN:
    def __init__(self) -> None:
        self.net = None

    def eval(self):
        pass
        # self.net.eval()

    def device(self):
        pass
        # self.net.to(torch.device("cuda"))
    
    def load_model(self):
        pass
    
    @torch.no_grad()
    def inference(self, img, sr_factor=2.0):
        img = F.interpolate(img, scale_factor=sr_factor, mode='bicubic', align_corners=False).clip(0, 1)
        return img
