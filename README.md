# Very Mix WebUI

## Introduction
Very Mix WebUI is a video enhancement tool, based on AI models, which achieving 2x, 4x, and 6x frame interpolation for videos, as well as 2x, and 4x upscaling.

## Supported Features
* [x] Frame Interpolation
    * [x] RIFE
    * [x] EMAVFI
* [x] Upscaling
    * [x] RLFN
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
conda create -n verymix-webui python=3.10 pytorch==1.12.0 cudatoolkit=11.3 -c pytorch -y
```

#### Install the Remaining Dependencies
```shell
conda activate verymix-webui
python -m pip --no-cache-dir install -r requirements-deploy-py3.10.txt
```

#### Download model files 
RIFE v4.6 | [flownet.pkl](https://github.com/hzwer/Practical-RIFE)

EMAVFI | [ours_small_t.pkl, ours_t.pkl](https://github.com/MCG-NJU/EMA-VFI) (please rename ours_small_t.pkl, ours_t.pkl to emavfi_s_t.pkl, emavfi_t.pkl respectively)

RLFN | [rlfn_s_x2.pth, rlfn_s_x4.pth](https://github.com/bytedance/RLFN/tree/main/model_zoo)

To be added:

RealESRGAN

ShuffleCUGAN

Video Restore

Place the model file in the appropriate folder, eg:

```shell
ckpt/
├── EMAVFI
│   ├── PutCheckpointsHere.txt
│   ├── emavfi_s_t.pkl
│   └── emavfi_t.pkl
├── RIFEv4.6
│   ├── PutCheckpointsHere.txt
│   └── flownet.pkl
```

## Usage

Start the web service:
```shell
export GRADIO_TEMP_DIR=./temp/gradio && python webui.py --config=config.yaml  # GRADIO_TEMP_DIR: Specify the file storage path
```

## Acknowledgements

- https://github.com/jhogsett/EMA-VFI-WebUI
- https://github.com/megvii-research/ECCV2022-RIFE
- https://github.com/MCG-NJU/EMA-VFI
- https://github.com/bytedance/RLFN
- https://github.com/xinntao/Real-ESRGAN
- https://gradio.app/