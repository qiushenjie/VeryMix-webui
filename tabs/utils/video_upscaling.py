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
from queue import Queue
from pathlib import Path
from typing import Callable

from models.utils.padder import InputPadder
from webui_utils.video_utils import ffmpegProcessor, VideoFrameIO


class Upscaling:
    def __init__(self, model, log_fn: Callable | None):
        self.model = model
        self.log_fn = log_fn
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def upscaling(self, args, progress=tqdm):
        # 当处理的数据为视频时，获取原始视频的帧率，帧数等信息；确定插帧倍数、输入视频路径等；打印插帧信息
        videoCapture = cv2.VideoCapture(str(args.video_input))
        h = videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        w = videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        videoCapture.release()

        args.w, args.h = int(w * args.sr_factor), int(h * args.sr_factor)
        args.fps = fps

        self.log('INFO: {}x{} to {}x{}.'.format(int(h), int(w), int(h * args.sr_factor), int(w * args.sr_factor)))

        video_name_wo_ext = args.video_input.stem
        args.video_output = '{}/{}.{}x{}.{}'.format(args.video_output, video_name_wo_ext, int(h * args.sr_factor), int(w * args.sr_factor), args.ext)

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
        divisor = max(16, int(16 / args.sr_factor))
        padder = InputPadder([int(args.h * args.sr_factor), int(args.w * args.sr_factor)], divisor=divisor, fp16=args.fp16)

        pbar = progress.tqdm(video_in, total=tot_frame, desc="progressing", unit=f" / {int(tot_frame)} frames")

        # 开始超分
        while True:
            frame = read_buffer.get()
            if frame is None:
                break
            if args.enhance_stop:
                self.log("ERROR: The video interpolation process was stopped by user.") 
                break

            frame = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.
            frame = padder.pad(frame)
            sr_frame = self.model.inference(frame, sr_factor=args.sr_factor)
            sr_frame = (sr_frame[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)

            write_buffer.put(sr_frame)
            pbar.update(1)

        while(not write_buffer.empty()):
            time.sleep(0.5)
        # pbar.close()
        if not video_out is None:
            video_out.stdin.close()

        if not args.enhance_stop:
            try:
                audio_input = Path(args.video_output).with_suffix('.aac')
                # extract audio
                os.system(f"{args.ffmpeg} -y -i {args.video_input} -vn -acodec copy {audio_input}")
                # merge audio and video
                os.system(f"{args.ffmpeg} -y -i {audio_input} -i {args.video_output} -c copy {args.video_output}")
                os.remove(audio_input)
            except:
                os.rename(args.video_output, Path(args.video_output).with_suffix(f'.noAudio.{args.ext}'))
                args.video_output = Path(args.video_output).with_suffix(f'.noAudio.{args.ext}')
                self.log("Warning: Create output video with no audio.")
            
        video_preview = Path(args.video_output).with_suffix('.demo.mp4')
        os.system(f"{args.ffmpeg} -y -i {args.video_output} -t 60 -c:v libx264 -crf 22 -vsync 0 {video_preview}")

        return video_preview, args.video_output

    def log(self, message):
        """Logging"""
        if self.log_fn:
            self.log_fn(message)