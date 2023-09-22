# Very Mix WebUI

## Introduction
Very Mix WebUI is a video enhancement tool, based on AI models, which achieving 2x, 4x, and 6x frame interpolation for videos, as well as 2x upscaling.

## Supported Features
* [x] Frame Interpolation
    * [x] RIFE
    * [x] EMAVFI
* [ ] Upscaling
    * [ ] RLFN
    * [ ] RealESRGAN
    * [ ] ShuffleCUGAN
* [ ] Restore

## Environment
```
Ubuntu 18.04
CUDA 11.3
CUDNN 8
Python 3.10.12 (Currently supporting versions not lower than 3.10)
PyTorch 1.12.0
Gradio 3.36.1
```

For other dependencies, please refer to
```
requirements.txt
```

### Installation

#### FFmpeg Set Up

A few features rely on FFmpeg being available on the system path

[Download FFmpeg](https://ffmpeg.org/download.html)

#### Miniconda Installation
It is recommended to use Miniconda.
```shell
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh
chmod u+x Miniconda3-py38_4.9.2-Linux-x86_64.sh
./Miniconda3-py38_4.9.2-Linux-x86_64.sh
```

#### Create a New Conda Environment
```shell
conda create -n video-enhance-webui python=3.10 pytorch==1.12.0 cudatoolkit=11.3 -c pytorch -y
```

#### Install the Remaining Dependencies
```shell
conda activate video-enhance-webui
python -m pip --no-cache-dir install -r requirements-deploy-py3.10.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
```

#### Download model files 
[Download RIFE v4.6](https://github.com/hzwer/Practical-RIFE)
[Download EMAVFI](https://github.com/MCG-NJU/EMA-VFI)
[Download RLFN]()
[Download RealESRGAN]()
[Download ShuffleCUGAN]()

## Usage

Start the web service:
```shell
export GRADIO_TEMP_DIR=./temp/gradio && python webui.py --config=config.yaml  # GRADIO_TEMP_DIR: Specify the file storage path
```

## Acknowledgements

Thanks! to the RIFE and EMA-VFI folks for their amazing AI frame interpolation tool
- https://github.com/megvii-research/ECCV2022-RIFE
- https://github.com/MCG-NJU/EMA-VFI

Thanks! to the Real-ESRGAN folks for their wonderful frame upscaling tool
- https://github.com/xinntao/Real-ESRGAN

Thanks! to the EMA-VFI-WebUI folks for their great UI, amazing tool, and for inspiring me to learn Gradio
- https://github.com/jhogsett/EMA-VFI-WebUI

Thanks to Gradio for their easy-to-use Web UI building tool and great docs
- https://gradio.app/
- https://github.com/gradio-app/gradio