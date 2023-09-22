"""Create Gradio markdown elements for guide markdown documents"""
import os
import gradio as gr

class WebuiTips:
    """Encapsulate logic to turn .md docs into Gradio markdown"""
    tips_path = "./guide"

    @classmethod
    def set_tips_path(cls, tips_path : str):
        """Point to the location of the tips directory"""
        cls.tips_path = tips_path

    @staticmethod
    def load_markdown(path : str, name : str):
        """Load a .md file and return it"""
        filepath = os.path.join(path, name + ".md")
        markdown = ""
        with open(filepath, encoding="utf-8") as file:
            markdown = file.read()
        return markdown

    video_interpolation = gr.Markdown(load_markdown(tips_path, "video_interpolation"))
    video_interpolation_ch = gr.Markdown(load_markdown(tips_path, "video_interpolation-ch"))
    video_upscaling = gr.Markdown(load_markdown(tips_path, "video_upscaling"))
    video_upscaling_ch = gr.Markdown(load_markdown(tips_path, "video_upscaling-ch"))
