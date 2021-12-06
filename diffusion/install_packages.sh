#! /bin/sh
set -e
eval "$($HOME/conda/bin/conda shell.bash hook)"
conda activate art

pip install lpips

if [ ! -d ../lib/CLIP ]; then
    git clone https://github.com/openai/CLIP.git ../lib/CLIP
    pip install -e ../lib/CLIP
fi

if [ ! -d ../lib/guided-diffusion ]; then
    git clone https://github.com/crowsonkb/guided-diffusion ../lib/guided-diffusion
    pip install -e ../lib/guided-diffusion
fi

mkdir weights; cd weights
curl -OL https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
curl -OL https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet.pth
curl -OL https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth
cd ..
