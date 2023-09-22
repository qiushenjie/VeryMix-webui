from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, '..')))

import importlib
from webui_utils.file_utils import scandir


# automatically scan and import dataset modules for registry
# scan all the files under the data folder with '_ui' in file names
data_folder = os.path.dirname(os.path.abspath(__file__))
dataset_filenames = [os.path.splitext(os.path.basename(v))[0] for v in scandir(data_folder) if v.endswith('_ui.py')]
# import all the dataset modules
_ui_modules = [importlib.import_module(f'tabs.{file_name}') for file_name in dataset_filenames]
