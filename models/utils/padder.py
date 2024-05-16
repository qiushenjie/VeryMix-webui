import torch.nn.functional as F


# ref:https://github.com/MCG-NJU/EMA-VFI/blob/75b6f6a889e695df875e103374040d47a4cfac7c/benchmark/utils/padder.py#L6
class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, dims, divisor=16, fp16=False):
        self.h, self.w = dims[-2:]
        self.fp16 = fp16
        ph = ((self.h - 1) // divisor + 1) * divisor
        pw = ((self.w - 1) // divisor + 1) * divisor
        self._pad = (0, pw - self.w, 0, ph - self.h)

    def pad(self, x):
        return F.pad(x, self._pad, mode='replicate').half() if self.fp16 else F.pad(x, self._pad, mode='replicate')
    
    def unpad(self, x):
        return x[..., :self.h, :self.w]