from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

"""Video Upscaling feature UI and event handlers"""
import shutil
import secrets
import tempfile
import gradio as gr
from pathlib import Path
from typing import Callable

from tabs.tab_base import TabBase
from tabs.utils.video_upscaling import Upscaling
from engine.upscaling_engine import UpscalingEngine

from webui_tips import WebuiTips
from webui_utils.simple_log import SimpleLog
from webui_utils.simple_icons import SimpleIcons
from webui_utils.simple_config import SimpleConfig

DEFAULT_TEMP_DIR = os.environ.get("GRADIO_TEMP_DIR") or str(Path(tempfile.gettempdir()) / "gradio")


class TaskConfig:
    def __init__(self):
        # codec setting
        self.w = None
        self.h = None
        self.ffmpeg = "ffmpeg"
        self.codec = "x264"
        self.fps = None
        self.crf = "22"

        # video file info
        self.video_input = None  # src video
        self.video_output = None  # dst video
        self.video_preview = None  # previewed dst video, less than 60s

        # task state
        self.enhance_stop = False  # upscaling stop or not
        
        # inference setting
        self.model_name = None
        self.engine = None
        self.sr_factor = 2.0
        self.fp16 = False
        

class SessionState(TaskConfig):
    def __init__(self):
        super().__init__()
        # task state
        self.task_id = None
        self.task_stop = None  # task stop or not

        # log
        self.log_obj = SimpleLog(verbose=False)


class VideoUpscaling(TabBase):
    """Encapsulates UI elements and events for the Frame Upscaling feature"""
    def __init__(self, base_config:SimpleConfig, log_fn:Callable):
        TabBase.__init__(self, base_config, log_fn)
        self.base_config = base_config

    def render_tab(self):
        
        """Render tab into UI"""
        with gr.Tab("Video Upscaling"):
            gr.HTML("AI-Based Video Upscaling Tool", elem_id="tabheading")

            # each time a new web page (session) is opened, a global state is assigned to the session, and this state is not shared between different users
            sr_session_state = gr.State(SessionState())

            with gr.Row():
                with gr.Column():
                    # setting component
                    with gr.Row():
                        model_name = gr.Dropdown(['RLFN', 'RealESRGAN', 'ShuffleCUGAN'], value="RLFN", label='Choose Model')  # 模型
                        sr_factor = gr.Slider(value=2.0, minimum=2.0, maximum=4.0, step=2.0, label='Upscaling Factor')  # 插帧倍数
                    with gr.Accordion("Setting", open=True):  # 设置
                        with gr.Row():
                            codec = gr.Dropdown(["x264", "x265"], value="x264", label='Codec')  # 视频编码器
                            crf = gr.Textbox(value="22", label="CRF", interactive=True)
                            ext = gr.Dropdown(["mp4", "mkv", "avi", "wmv", "flv", "m4v", "mov"], value="mp4", label="Output Extension")  # 视频格式

                # video input/previewed/output
                with gr.Column():
                    with gr.Column():
                        video_upload = gr.File(type="filepath", label="Input Video", file_types=[".mp4", ".mkv", ".avi", ".wmv", ".flv", ".m4v", ".mov",".ts", ".webm", ".rmvb"])  # 上传视频
                        dst_video_preview = gr.Video(label="Previewed Video", format="mp4", visible=False)  # 视频预览
                        video_output = gr.File(type="filepath", label="Video Download", file_count="multiple", visible=False)  # 视频下载

                        with gr.Row():
                            run_btn = gr.Button("Run", variant="primary", interactive=False)
                            stop_btn = gr.Button("Stop", variant="primary", interactive=False)

                    # log component
                    max_lines = self.base_config.logviewer_settings["max_lines"]
                    with gr.Accordion(SimpleIcons.SCROLL + " Logs", open=False):
                        with gr.Row():
                            log_text = gr.Textbox(max_lines=max_lines, placeholder="Press Refresh Log", label="Logs",
                                                interactive=False, elem_id="logviewer")
                        with gr.Row():
                            refresh_button = gr.Button(value="Refresh Log", variant="primary")
                            clear_button = gr.Button(value="Clear Log")
            
            # guide component
            with gr.Accordion(SimpleIcons.TIPS_SYMBOL + " Guide", open=False):
                with gr.Tab("English"):
                    WebuiTips.video_upscaling.render()
                with gr.Tab("中文"):
                    WebuiTips.video_upscaling_ch.render()


        ########################################### event listen ###########################################
        # refresh button state after refresh upload video
        video_upload.change(lambda video_upload, sr_session_state:
                            (
                             gr.update(interactive=((video_upload != None) and (sr_session_state.task_stop == None))),
                             gr.update(interactive=(sr_session_state.task_stop != None))
                            ),
                            inputs=[video_upload, sr_session_state],
                            outputs=[run_btn, stop_btn])

        # click "Run": init session state -> run enhancement task -> refresh button state during enhancing
        run_btn\
            .click(self.on_click_run, inputs=None, outputs=[dst_video_preview, video_output, run_btn, stop_btn])\
            .success(
                self.init_state,
                inputs=[
                    sr_session_state,
                    model_name,
                    sr_factor,
                    codec, crf, ext,
                    video_upload,
                    ],
                outputs=[sr_session_state])\
            .success(self.run_enhance, inputs=[sr_session_state], outputs=[sr_session_state, dst_video_preview, video_output, run_btn, stop_btn])\
            .success(lambda :  # 注释:和 gradio3.36.1 版本不同的是，在 gradio4 中，如果 button 作为参数传入函数中，在函数运行过程中，button 的状态是不可更改的，直到函数运行完成。
                     (
                        gr.update(interactive=True),
                        gr.update(interactive=False)
                     ),
                     inputs=None,
                     outputs=[run_btn, stop_btn])

        # click "Stop": refresh button state -> cancel enhancement task -> output the finished part
        stop_btn\
            .click(self.on_click_stop, inputs=None, outputs=[run_btn, stop_btn])\
            .success(self.stop_state_change, inputs=[sr_session_state], outputs=[sr_session_state])

        # refresh log/clear log
        refresh_button.click(self.on_click_refresh_log, inputs=[sr_session_state], outputs=log_text)
        clear_button.click(self.on_click_clear_log, inputs=[sr_session_state], outputs=log_text)

    def on_click_run(self):
        return \
            gr.update(value=None, visible=False),\
            gr.update(value=None, visible=True),\
            gr.update(interactive=False),\
            gr.update(interactive=True)

    def on_click_stop(self):
        """
        1. Disable Run and Stop button immediately
        2. Enable Run button after Very Mix process finished
        """
        return gr.update(interactive=False), gr.update(interactive=False)
    
    def on_click_refresh_log(self, session_state):
        messages = session_state.log_obj.messages
        return "\n".join(messages)

    def on_click_clear_log(self, session_state):
        session_state.log_obj.reset()
        return gr.update(value="", placeholder="Press Refresh Log")

    def init_state(self,
                    sr_session_state,
                    model_name,
                    sr_factor,
                    codec, crf, ext,
                    video_upload,
                   ):
        if video_upload:
            sr_session_state.video_input = video_upload.name
        else:
            sr_session_state.video_input = None

        sr_session_state.task_stop = sr_session_state.enhance_stop = False
        sr_session_state.model_name = model_name
        sr_session_state.sr_factor = sr_factor
        sr_session_state.codec = codec
        sr_session_state.crf = float(crf)
        sr_session_state.ext = ext
        
        return sr_session_state

    def stop_state_change(self, session_state):
        """Function to handle stop button is clicked.
        ref:https://github.com/XingangPan/DragGAN/blob/e77f69d665c22ad0fdf5c4c1be4183556ae44387/visualizer_drag_gradio.py#L672C9-L672C22
        send a stop signal by set session_state.task_stop as None
        """
        session_state.task_stop = None
        session_state.enhance_stop = True
        return session_state

    def run_enhance(self, sr_session_state, progress=gr.Progress()):
        # model init
        sr_session_state.engine = UpscalingEngine(self.base_config.gpu_ids, sr_session_state.sr_factor)
        sr_session_state.engine.load(sr_session_state.model_name)
        sr = Upscaling(sr_session_state.engine.model, sr_session_state.log_obj.log)

        try:
            sr_session_state.log_obj.log(f"INFO: Preparing video...")

            sr_session_state.video_input = Path(sr_session_state.video_input)

            # create video saving dir by task ID
            base_output_path = self.base_config.directories["output_video_upscaling"]
            runId = sr_session_state.video_input.parent.stem[:6]
            output_dir = os.path.join(base_output_path, runId)
            os.makedirs(output_dir, exist_ok=True)
            
            sr_session_state.video_output = output_dir
            sr_session_state.task_id = runId
            sr_session_state.log_obj.log(f"INFO: Get runing task {sr_session_state.task_id}")

            sr_session_state.log_obj.log(f"INFO: Start upscaling...")
            
            # run upscaling
            preview_path, download_path = sr.upscaling(sr_session_state, progress)

            if not os.path.isfile(preview_path):
                preview_path = None
                sr_session_state.log_obj.log(f"INFO: Video demo run failed.")
            else:
                sr_session_state.log_obj.log(f"INFO: Video demo run succeed.")
            if not os.path.isfile(download_path):
                download_path = None
                sr_session_state.log_obj.log(f"INFO: Video upscaling run failed.")
            else:
                sr_session_state.log_obj.log(f"INFO: Video upscaling run succeed.")

        except Exception as e:
            sr_session_state.log_obj.log(f"ERROR: Video upscaling failed...\t{e}")
            preview_path, download_path = None, None
        sr_session_state.log_obj.log("")  # task completed, log line feed

        # model release
        sr_session_state.engine.release()

        # update current session state
        sr_session_state.video_output = str(download_path)
        sr_session_state.video_preview = str(preview_path)
        sr_session_state.task_stop = sr_session_state.enhance_stop = None

        return \
            sr_session_state,\
            gr.update(value=str(preview_path), visible=True),\
            gr.update(value=str(download_path), visible=True),\
            gr.update(interactive=True),\
            gr.update(interactive=False)
