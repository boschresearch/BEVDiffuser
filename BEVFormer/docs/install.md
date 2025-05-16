# Step-by-step installation instructions

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n bevdiffuser python=3.8 -y
conda activate bevdiffuser
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```
Note: we use torch==1.10.0 to be compatible with diffusion related packages. 

**c. Install gcc>=5 in conda env (optional).**
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
#  pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
pip install -v -e .
```

**f. Install Detectron2 and Timm.**
```shell
pip install einops fvcore seaborn iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 setuptools==59.5.0
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**g. Install packages for diffusion model.**
```shell
pip install accelerate==0.20.3 diffusers==0.20.0 transformers==4.29.2 safetensors==0.3.0 wandb==0.15.8 datasets ftfy tensorboard Jinja2 tabulate scipy yapf==0.40.1
```


**h. Clone BEVDiffuser.**
```
git clone https://github.com/Xin-Ye-1/BEVDiffuser.git
```

**i. Prepare pretrained models.**
```shell
cd BEVDiffuser/BEVFormer
mkdir ckpts

cd ckpts & wget https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth
```

