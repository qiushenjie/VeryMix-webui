import torch
from functools import partial
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from models.EMAVFI.config import *
from models.EMAVFI.refine import *
from models.EMAVFI.feature_extractor import feature_extractor
from models.EMAVFI.flow_estimation import MultiScaleFlow as flow_estimation


'''==========Model config=========='''
def init_model_config(F=32, W=7, depth=[2, 2, 2, 4, 4]):
    '''This function should not be modified'''
    return {
        'embed_dims': [F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims': [0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads': [8*F//32, 16*F//32],
        'mlp_ratios': [4, 4],
        'qkv_bias': True,
        'norm_layer': partial(nn.LayerNorm, eps=1e-6),
        'depths': depth,
        'window_sizes': [W, W]
    }, {
        'embed_dims': [F, 2*F, 4*F, 8*F, 16*F],
        'motion_dims': [0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths': depth,
        'num_heads': [8*F//32, 16*F//32],
        'window_sizes': [W, W],
        'scales': [4, 8, 16],
        'hidden_dims': [4*F, 4*F],
        'c': F
    }


class EMAVFI:
    def __init__(self, name="emavfi_s", local_rank=-1):
        backbonetype, multiscaletype = feature_extractor, flow_estimation
        if name == "emavfi_s":
            backbonecfg, multiscalecfg = init_model_config(**MODEL_CONFIG_S['MODEL_ARCH'])
            self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
            self.name = MODEL_CONFIG_S['LOGNAME']
        else:
            backbonecfg, multiscalecfg = init_model_config(**MODEL_CONFIG['MODEL_ARCH'])
            self.net = multiscaletype(backbonetype(**backbonecfg), **multiscalecfg)
            self.name = MODEL_CONFIG['LOGNAME']
        self.device()

        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[
                           local_rank], output_device=local_rank)

    def eval(self):
        self.net.eval()

    def device(self):
        self.net.to(torch.device("cuda"))

    def load_model(self, name=None, rank=0):
        def convert(param):
            return {
            k.replace("module.", ""): v
                for k, v in param.items()
                if "module." in k and 'attn_mask' not in k and 'HW' not in k
            }
        if rank <= 0 :
            if name is None:
                name = self.name
            # state_dict = convert(torch.load(f'./ckpt/EMAVFI/{name}.pkl'))
            state_dict = convert(torch.load(f'./ckpt/EMAVFI/{name}_t.pkl'))
            self.net.load_state_dict(state_dict)

    def save_model(self, rank=0):
        if rank == 0:
            # torch.save(self.net.state_dict(), f'./ckpt/EMAVFI/{self.name}.pkl')
            torch.save(self.net.state_dict(), f'./ckpt/EMAVFI/{self.name}_t.pkl')

    @torch.no_grad()
    def hr_inference(self, img0, img1, timestep=0.5, scale_factor=0.5, TTA=False, fast_TTA=False):
        '''
        Infer with scale_factor flow
        Noting: return BxCxHxW
        '''
        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            imgs_down = F.interpolate(
                imgs, scale_factor=scale_factor, mode="bilinear", align_corners=False)

            flow, mask = self.net.calculate_flow(imgs_down, timestep)

            flow = F.interpolate(flow, scale_factor=1/scale_factor,
                                 mode="bilinear", align_corners=False) * (1/scale_factor)
            mask = F.interpolate(
                mask, scale_factor=1/scale_factor, mode="bilinear", align_corners=False)

            af, _ = self.net.feature_bone(img0, img1)
            pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
            return pred

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds = infer(input)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        if TTA == False:
            return infer(imgs)
        else:
            return (infer(imgs) + infer(imgs.flip(2).flip(3)).flip(2).flip(3)) / 2

    @torch.no_grad()
    def inference(self, img0, img1, timestep=0.5, scale_factor=1.0, TTA=False, fast_TTA=False):
        imgs = torch.cat((img0, img1), 1)
        '''
        Noting: return BxCxHxW
        '''
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            _, _, _, preds = self.net(input, timestep=timestep)
            return (preds[0] + preds[1].flip(1).flip(2)).unsqueeze(0) / 2.

        _, _, _, pred = self.net(imgs, timestep=timestep)
        if TTA == False:
            return pred
        else:
            _, _, _, pred2 = self.net(imgs.flip(2).flip(3), timestep=timestep)
            return (pred + pred2.flip(2).flip(3)) / 2

    @torch.no_grad()
    def multi_inference(self, img0, img1, time_list=[], scale_factor=1.0, TTA=False, fast_TTA=False):
        '''
        Run backbone once, get multi frames at different timesteps
        Noting: return a list of [CxHxW]
        '''
        assert len(time_list) > 0, 'Time_list should not be empty!'

        def infer(imgs):
            img0, img1 = imgs[:, :3], imgs[:, 3:6]
            af, mf = self.net.feature_bone(img0, img1)
            imgs_down = None
            if scale_factor != 1.0:
                imgs_down = F.interpolate(
                    imgs, scale_factor=scale_factor, mode="bilinear", align_corners=False)
                afd, mfd = self.net.feature_bone(
                    imgs_down[:, :3], imgs_down[:, 3:6])

            pred_list = []
            for timestep in time_list:
                if imgs_down is None:
                    flow, mask = self.net.calculate_flow(
                        imgs, timestep, af, mf)
                else:
                    flow, mask = self.net.calculate_flow(
                        imgs_down, timestep, afd, mfd)
                    flow = F.interpolate(
                        flow, scale_factor=1/scale_factor, mode="bilinear", align_corners=False) * (1/scale_factor)
                    mask = F.interpolate(
                        mask, scale_factor=1/scale_factor, mode="bilinear", align_corners=False)

                pred = self.net.coraseWarp_and_Refine(imgs, af, flow, mask)
                pred_list.append(pred)

            return pred_list

        imgs = torch.cat((img0, img1), 1)
        if fast_TTA:
            imgs_ = imgs.flip(2).flip(3)
            input = torch.cat((imgs, imgs_), 0)
            preds_lst = infer(input)
            return [(preds_lst[i][0] + preds_lst[i][1].flip(1).flip(2))/2 for i in range(len(time_list))]

        preds = infer(imgs)
        if TTA is False:
            return [preds[i][0] for i in range(len(time_list))]
        else:
            flip_pred = infer(imgs.flip(2).flip(3))
            return [(preds[i][0] + flip_pred[i][0].flip(1).flip(2))/2 for i in range(len(time_list))]
