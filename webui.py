"""Very-Mix-WebUI Application"""
import os
import time
import signal
import argparse
from typing import Callable

from webui_tips import WebuiTips
from webui_utils.simple_log import SimpleLog
from create_ui import create_ui
from webui_utils.simple_config import SimpleConfig
from webui_utils.file_utils import create_directories


def main():
    """Run the application"""
    parser = argparse.ArgumentParser(description='Very Max WebUI')
    parser.add_argument("--config_path", type=str, default="config.yaml",
        help="path to config YAML file")
    parser.add_argument("--verbose", dest="verbose", default=False, action="store_true",
        help="Show extra details")
    args = parser.parse_args()
    log = SimpleLog(args.verbose)
    base_config = SimpleConfig(args.config_path).config_obj()
    create_directories(base_config.directories)
    WebUI(base_config, log).start()

class WebUI:
    """Top-Level application logic"""
    def __init__(self,
                    base_config : SimpleConfig,
                    log : SimpleLog):
        self.base_config = base_config
        self.log = log
        self.restart = False
        self.prevent_inbrowser = False

    def start(self):
        """Create the UI and start the event loop"""
        WebuiTips.set_tips_path(self.base_config.user_interface["tips_path"])
        while True:
            print("\nStarting Very-Mix-WEB")
            app = create_ui(self.base_config, self.log, self.restart_app)
            app.queue(concurrency_count=5, max_size=8)
            app.launch(inbrowser = self.base_config.auto_launch_browser and not self.prevent_inbrowser,
                        server_name = self.base_config.server_name,
                        server_port = self.base_config.server_port,
                        prevent_thread_lock=True)
            # after initial launch, disable inbrowser for subsequent restarts
            self.prevent_inbrowser = True
            self.wait_on_server(app)
            print("\n--- Restarting\n")

    def restart_app(self):
        """Signal to the event loop to restart the application"""
        self.restart = True

    def wait_on_server(self, app):
        """Restart application if signal is set"""
        while True:
            time.sleep(0.5)
            if self.restart:
                self.restart = False
                time.sleep(0.5)
                app.close()
                time.sleep(0.5)
                break

def sigint_handler(sig, frame):
    """Make the program just exit at ctrl+c without waiting for anything"""
    print(f'Interrupted with signal {sig} in {frame}')
    os._exit(0) #pylint: disable=protected-access
signal.signal(signal.SIGINT, sigint_handler)

if __name__ == '__main__':
    main()
