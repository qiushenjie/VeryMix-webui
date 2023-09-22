"""Functions for dealing with video using FFmpeg"""
import os
import cv2
import glob
import ffmpeg
import subprocess
import skvideo.io
import numpy as np
from typing import Callable
from ffmpy import FFmpeg, FFprobe
from .file_utils import split_filepath
from .image_utils import gif_frame_count


QUALITY_NEAR_LOSSLESS = 17
QUALITY_SMALLER_SIZE = 28
QUALITY_DEFAULT = 23


def determine_pattern(input_path : str):
    """Determine the FFmpeg wildcard pattern needed for a set of files"""
    files = sorted(glob.glob(os.path.join(input_path, "*.png")))
    first_file = files[0]
    file_count = len(files)
    num_width = len(str(file_count))
    _, name_part, ext_part = split_filepath(first_file)
    return f"{name_part[:-num_width]}%0{num_width}d{ext_part}"


def PNGtoMP4(input_path : str, # pylint: disable=invalid-name
            filename_pattern : str,
            frame_rate : int,
            output_filepath : str,
            crf : int = QUALITY_DEFAULT):
    """Encapsulate logic for the PNG Sequence to MP4 feature"""
    # if filename_pattern is "auto" it uses the filename of the first found file
    # and the count of file to determine the pattern, .png as the file type
    # ffmpeg -framerate 60 -i .\upscaled_frames%05d.png -c:v libx264 -r 60  -pix_fmt yuv420p
    #   -crf 28 test.mp4    if filename_pattern == "auto":
    filename_pattern = determine_pattern(input_path)
    ffcmd = FFmpeg(
        inputs= {os.path.join(input_path, filename_pattern) : f"-framerate {frame_rate}"},
        outputs={output_filepath : f"-c:v libx264 -r {frame_rate} -pix_fmt yuv420p -crf {crf}"},
        global_options="-y")
    cmd = ffcmd.cmd
    ffcmd.run()
    return cmd


# ffmpeg -y -i frames.mp4 -filter:v fps=25 -pix_fmt rgba -start_number 0 output_%09d.png
# ffmpeg -y -i frames.mp4 -filter:v fps=25 -start_number 0 output_%09d.png
def MP4toPNG(input_path : str,  # pylint: disable=invalid-name
            filename_pattern : str,
            frame_rate : int,
            output_path : str,
            start_number : int = 0):
    """Encapsulate logic for the MP4 to PNG Sequence feature"""
    ffcmd = FFmpeg(inputs= {input_path : None},
        outputs={os.path.join(output_path, filename_pattern) :
            f"-filter:v fps={frame_rate} -start_number {start_number}"},
        global_options="-y")
    cmd = ffcmd.cmd
    ffcmd.run()
    return cmd

# making a high quality GIF from images requires first creating a color palette,
# then supplying it to the conversion command
# https://stackoverflow.com/questions/58832085/colors-messed-up-distorted-when-making-a-gif-from-png-files-using-ffmpeg

# ffmpeg -i gifframes_%02d.png -vf palettegen palette.png
def PNGtoPalette(input_path : str, # pylint: disable=invalid-name
                filename_pattern : str,
                output_filepath : str):
    """Create a palette from a set of PNG files to feed into animated GIF creation"""
    if filename_pattern == "auto":
        filename_pattern = determine_pattern(input_path)
    ffcmd = FFmpeg(inputs= {os.path.join(input_path, filename_pattern) : None},
                outputs={output_filepath : "-vf palettegen"},
                global_options="-y")
    cmd = ffcmd.cmd
    ffcmd.run()
    return cmd


def PNGtoGIF(input_path : str, # pylint: disable=invalid-name
            filename_pattern : str,
            output_filepath : str,
            frame_rate : int):
    """Encapsulates logic for the PNG sequence to GIF feature"""
    # if filename_pattern is "auto" it uses the filename of the first found file
    # and the count of file to determine the pattern, .png as the file type
    # ffmpeg -i gifframes_%02d.png -i palette.png -lavfi paletteuse video.gif
    # ffmpeg -framerate 3 -i image%01d.png video.gif
    if filename_pattern == "auto":
        filename_pattern = determine_pattern(input_path)
    output_path, base_filename, _ = split_filepath(output_filepath)
    palette_filepath = os.path.join(output_path, base_filename + "-palette.png")
    palette_cmd = PNGtoPalette(input_path, filename_pattern, palette_filepath)

    ffcmd = FFmpeg(inputs= {
            os.path.join(input_path, filename_pattern) : f"-framerate {frame_rate}",
            palette_filepath : None},
        outputs={output_filepath : "-lavfi paletteuse"},
        global_options="-y")
    cmd = ffcmd.cmd
    ffcmd.run()
    return "\n".join([palette_cmd, cmd])


def GIFtoPNG(input_path : str, # pylint: disable=invalid-name
            output_path : str,
            start_number : int = 0):
    """Encapsulates logic for the GIF to PNG Sequence feature"""
    # ffmpeg -y -i images\example.gif -start_number 0 gifframes_%09d.png
    _, base_filename, extension = split_filepath(input_path)

    if extension.lower() == ".gif":
        frame_count = gif_frame_count(input_path)
    elif extension.lower() == ".mp4":
        frame_count = mp4_frame_count(input_path)
    else:
        # assume an arbitrarily high frame count to ensure a wide index
        frame_count = 1_000_000

    num_width = len(str(frame_count))
    filename_pattern = f"{base_filename}%0{num_width}d.png"
    ffcmd = FFmpeg(inputs= {input_path : None},
        outputs={os.path.join(output_path, filename_pattern) : f"-start_number {start_number}"},
        global_options="-y")
    cmd = ffcmd.cmd
    ffcmd.run()
    return cmd


def mp4_frame_count(input_path : str) -> int:
    """Using FFprobe to determine MP4 frame count"""
    # ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -print_format default=nokey=1:noprint_wrappers=1 Big_Buck_Bunny_1080_10s_20MB.mp4
    ff = FFprobe(inputs= {input_path : "-count_frames -show_entries stream=nb_read_frames -print_format default=nokey=1:noprint_wrappers=1"})
    return ff.run()


def probe_video_info(path, ffprobe="ffprobe"):
    args = [ffprobe, "-select_streams", "v", "-count_packets", "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets,bit_rate", "-of", "json", path]
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode != 0:
        return None
    j = json.loads(out.decode('utf-8'))
    width = int(j["streams"][0]["width"])
    height = int(j["streams"][0]["height"])
    fps = float(eval(j["streams"][0]["r_frame_rate"]))
    nb_frames = int(j["streams"][0]["nb_read_packets"])
    bit_rate = int(j["streams"][0]["bit_rate"]) / 1000
    return width, height, fps, nb_frames, bit_rate


class ffmpegProcessor:
    def __init__(self, args, log_fn : Callable | None):
        self.cmd = args.ffmpeg
        self.log_fn = log_fn
        self.codec_name = {
            "x264": "libx264",
            "x265": "libx265",
            "biliavc": "libbiliavc",
            "bilihevc": "libbilihevc",
        }

    def input_codec_dict(self, crf, codec="bilihevc", **kwargs):
        return {
                "vf": "null",
                "c:v": self.codec_name[codec],
        }

    def output_codec_dict(self, crf, codec="bilihevc", **kwargs):
        return {
                "c:v": self.codec_name[codec],
                "crf": crf,
                "preset": "veryfast",
                "vsync": 0,
                "color_range": "tv",
                "colorspace": "bt709",
                "color_trc": "bt709",
                "color_primaries": "bt709",
                "pix_fmt": "yuv420p",        
        }

    # 视频解码：方法1，解码慢，但全流程快
    def video_in1(self, args):
        return skvideo.io.vreader(args.video_input)

    # 视频解码：方法2，解码快，但全流程慢，遇到长视频容易崩溃，暂未解决
    def video_in2(self, args):
        video = (
            ffmpeg
            .input(args.video_input)  # 不对input做任何处理，默认只有一条视频流
            .output('pipe:', format='rawvideo', pix_fmt='rgb24')  # 指定输出数据格式为rawvideo
            .run_async(cmd=self.cmd, pipe_stdout=True)
        )
        return video
    
    # 视频编码
    def video_out(self, args):
        video = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(args.w, args.h), framerate=int(np.round(args.fps)))
            .output(args.video_output, **self.output_codec_dict(args.crf, args.codec))
            .overwrite_output()
            .run_async(cmd=self.cmd, pipe_stdin=True)
        )
        
        return video
    
    # 音视频合成
    def merge(self, args):
        video_path_wo_ext, ext = os.path.splitext(args.video_output)
        video_path_no_audio = f"{video_path_wo_ext}_with_audio{ext}"
        os.rename(args.video_output, video_path_no_audio)
        try:
            ffmpeg.\
                output(ffmpeg.input(args.video_input).audio, ffmpeg.input(video_path_no_audio).video, args.video_output, vcodec='copy', acodec='copy').\
                    overwrite_output().run()
            os.remove(video_path_no_audio)
        except Exception as e:
            self.log("ERROR: Audio transfer failed, interpolated video will have no audio.")
            os.rename(video_path_no_audio, args.video_output)

    def log(self, message):
        """Logging"""
        if self.log_fn:
            self.log_fn(message)


class VideoFrameIO:
    def __init__(self):
        pass

    def build_read_buffer(self, args, read_buffer, video_in):
        # # for ffmpeg video in
        # try:
        #     while True:
        #         frame = video_in.stdout.read(args.h * args.w * 3)
        #         if not frame:
        #             break
        #         frame = (np.frombuffer(frame, np.uint8)).reshape([args.h, args.w, 3])
        #         read_buffer.put(frame)
                
        # except:
        #     pass
        # read_buffer.put(None)
        
        # for skvideo video in
        try:
            for frame in video_in:
                read_buffer.put(frame)
        except:
            pass
        read_buffer.put(None)


    def clear_write_buffer(self, args, write_buffer, video_out):
        cnt = 0
        while True:
            frame = write_buffer.get()
            if frame is None:
                break
            frame = np.ascontiguousarray(frame)
            frame = frame.astype(np.uint8).tobytes()
            if not video_out.stdin.closed:
                video_out.stdin.write(frame)
                cnt += 1