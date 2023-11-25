"""Base class for UI tabs"""
from typing import Callable
from webui_utils.simple_log import SimpleLog
from webui_utils.simple_config import SimpleConfig


class TabBase():
    """Shared UI tab methods"""
    def __init__(self,
                base_config : SimpleConfig,
                log_fn : Callable):
        self.base_config = base_config
        self.log_fn = log_fn

    def log(self, message : str):
        """Logging"""
        self.log_fn(message)
