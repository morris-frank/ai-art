#! /bin/sh
set -e

eval "$($HOME/conda/bin/conda shell.bash hook)"

env_name=latents
conda deactivate
conda env remove --name $env_name
conda create -n $env_name python=3.9 -y
conda activate $env_name

#  Log in and out
exit()


conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install mamba -c conda-forge -y
mamba install -c conda-forge pytorch-lightning kornia ftfy rich pandas scipy ipython omegaconf regex ipdb -y
pip install lpips

mkdir lib
# git clone https://github.com/CompVis/taming-transformers  lib/taming-transformers
git clone https://github.com/openai/CLIP.git lib/CLIP
pip install -e ./lib/CLIP
git clone https://github.com/crowsonkb/guided-diffusion lib/guided-diffusion
pip install -e ./lib/guided-diffusion

mkdir weights; cd weights
curl -OL https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
curl -OL https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet.pth
curl -OL https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth
cd ..