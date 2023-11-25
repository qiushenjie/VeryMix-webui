from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '../..')))

import re
import cv2
import time
import tqdm
import torch
import _thread
import numpy as np
from tqdm import tqdm
from shutil import move
from queue import Queue
from pathlib import Path
from typing import Callable
from torch.nn import functional as F

from models.utils.padder import InputPadder
from models.utils.pytorch_msssim import ssim_matlab
from webui_utils.video_utils import ffmpegProcessor, VideoFrameIO


def frame_downscale(frame, scale):
    if scale < 1.0 and scale >= 0.5:
        scale_frame = F.interpolate(frame, scale_factor=scale, mode='bilinear', align_corners=False)
    else:
        scale_frame = frame
    return scale_frame


def frame_upscale(frame, scale):
    if scale < 1.0 and scale >= 0.5:
        scale_frame = F.interpolate(frame, scale_factor=1/scale, mode='bicubic', align_corners=False).clip(0, 1)
    else:
        scale_frame = frame
    return scale_frame

class Interpolate:
    def __init__(self, model, log_fn: Callable | None):
        self.model = model
        self.log_fn = log_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def interpolate(self, args, progress=tqdm):
        # 当处理的数据为视频时，获取原始视频的帧率，帧数等信息；确定插帧倍数、输入视频路径等；打印插帧信息
        videoCapture = cv2.VideoCapture(str(args.video_input))
        h = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()

        args.w, args.h = int(w), int(h)
        args.fps = fps * args.interpolation_factor

        self.log('INFO: {}FPS to {}FPS.'.format(fps, args.fps))

        video_name_wo_ext = args.video_input.stem
        args.video_output = '{}/{}.{}fps.{}'.format(args.video_output, video_name_wo_ext, int(np.round(args.fps)), args.ext)

        # 准备插帧的输入输出视频，并处理成队列
        ffmpeg_processor = ffmpegProcessor(args, self.log_fn)
        video_in = ffmpeg_processor.video_in1(args)
        video_out = ffmpeg_processor.video_out(args)

        video_frame_io = VideoFrameIO()
        read_buffer  = Queue(maxsize=100)
        write_buffer = Queue(maxsize=100)
        _thread.start_new_thread(video_frame_io.build_read_buffer,  (args, read_buffer, video_in))
        _thread.start_new_thread(video_frame_io.clear_write_buffer, (args, write_buffer, video_out))

        # 初始化padding，padding成满足模型推理的尺度
        divisor = max(16, int(16 / args.flow_scale))
        padder = InputPadder([int(args.h * args.frame_scale), int(args.w * args.frame_scale)], divisor=divisor, fp16=args.fp16)

        temp = None # save lastframe when processing static frame

        lastframe = read_buffer.get()
        I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.

        scale_I1 = frame_downscale(I1.clone(), args.frame_scale)
        scale_I1 = padder.pad(scale_I1)
        I1_small = F.interpolate(scale_I1, (64, 64), mode='bilinear', align_corners=False)

        pbar = progress.tqdm(video_in, total=tot_frame * args.interpolation_factor, desc="progressing", unit=f" / {int(tot_frame * args.interpolation_factor)} frames")
        
        # 开始插帧
        while True:
            if args.enhance_stop:
                self.log("ERROR: The video interpolation process was stopped by user.") 
                break
            if temp is not None:
                frame = temp
                temp = None
            else:
                frame = read_buffer.get()
            if frame is None:
                break
            I0 = I1
            scale_I0 = scale_I1

            I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
            scale_I1 = frame_downscale(I1.clone(), args.frame_scale)
            scale_I1 = padder.pad(scale_I1)

            I0_small = I1_small
            I1_small = F.interpolate(scale_I1, (64, 64), mode='bilinear', align_corners=False)

            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

            break_flag = False
            if ssim == 1:
                output = []
                for i in range(args.interpolation_factor - 1):
                    output.append(I0)

            elif ssim > args.skip_threshold:
                frame = read_buffer.get()  # read a new frame
                if frame is None:
                    break_flag = True
                    frame = lastframe
                else:
                    temp = frame
                Imids = []
                for i in range(args.interpolation_factor - 1):
                    timestep = torch.tensor((i + 1) * 1. / (args.interpolation_factor), device=self.device).clone().detach()
                    Imid = self.model.inference(scale_I0, scale_I1, timestep=timestep, scale_factor=args.flow_scale, TTA=args.TTA)
                    Imid = padder.unpad(Imid)
                    Imid = frame_upscale(Imid, args.frame_scale)
                    Imids.append(Imid)

                Imid_small = F.interpolate(Imids[(args.interpolation_factor - 1) // 2], (64, 64), mode='bilinear', align_corners=False)
                ssim_interp = min(ssim_matlab(I0_small[:, :3], Imid_small[:, :3]), ssim_matlab(I1_small[:, :3], Imid_small[:, :3]))
                frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)

                output = []
                if ssim_interp < args.bad_threshold:
                    for i in range(args.interpolation_factor - 1):
                        output.append(I1)

                else:
                    for mid in Imids:
                        output.append(mid)
            
            else:
                output = []
                for i in range(args.interpolation_factor - 1):
                    output.append(I0)

            write_buffer.put(lastframe)
            for mid in output:
                mid =(mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)
                write_buffer.put(mid)

            pbar.update(1 * args.interpolation_factor)
            lastframe = frame
            if break_flag:
                break

        write_buffer.put(lastframe)

        while(not write_buffer.empty()):
            time.sleep(0.5)
        # pbar.close()
        if not video_out is None:
            video_out.stdin.close()

        if not args.enhance_stop:
            no_audio_video_output = Path(args.video_output).with_suffix(f'.noAudio.{args.ext}')
            tmp_video_output = Path(args.video_output).with_suffix(f'.tmp.{args.ext}')
            audio_input = Path(args.video_output).with_suffix('.aac')
            
            # extract audio
            os.system(f"{args.ffmpeg} -y -i {args.video_input} -vn -acodec copy {audio_input}")
            # merge audio and video
            if os.path.exists(audio_input):
                os.system(f"{args.ffmpeg} -y -i {audio_input} -i {args.video_output} -c copy {tmp_video_output}")
                os.remove(args.video_output)
                os.remove(audio_input)
                move(tmp_video_output, args.video_output)
            else:
                move(args.video_output, no_audio_video_output)
                args.video_output = no_audio_video_output
                self.log("Warning: Create output video with no audio.")
            
        video_preview = Path(args.video_output).with_suffix('.demo.mp4')
        os.system(f"{args.ffmpeg} -y -i {args.video_output} -t 60 -c:v libx264 -crf 22 -vsync 0 {video_preview}")

        return video_preview, args.video_output

    def log(self, message):
        """Logging"""
        if self.log_fn:
            self.log_fn(message)