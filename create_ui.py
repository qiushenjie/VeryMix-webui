"""Create the Gradio UI elements"""
import gradio as gr
from typing import Callable

from webui_utils.simple_log import SimpleLog
from webui_utils.simple_icons import SimpleIcons
from webui_utils.simple_config import SimpleConfig

# inport ui
from tabs.video_interpolation_ui import VideoInterpolation
from tabs.video_upscaling_ui import VideoUpscaling


def create_ui(config : SimpleConfig,
              log : SimpleLog,
              restart_fn : Callable):
    """Construct the Gradio Blocks UI"""

    app_header = gr.HTML(SimpleIcons.CLAPPER + " VeryMix WebUI", elem_id="appheading")
    sep = '  •  '
    footer = (SimpleIcons.COPYRIGHT + ' VeryMix' +
        sep + '<a href="https://gradio.app">Gradio</a>')
    app_footer = gr.HTML(footer, elem_id="footer")

    with gr.Blocks(analytics_enabled=False,
                    title="Very Mix Web UI",
                    theme=gr.themes.Default(font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "sans-serif"]),
                    css=config.user_interface["css_file"]) as app:
        if config.user_interface["show_header"]:
            app_header.render()
        VideoInterpolation(config, log.log).render_tab()
        VideoUpscaling(config, log.log).render_tab()

        if config.user_interface["show_header"]:
            app_footer.render()
    return app
