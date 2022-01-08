#! /bin/sh
env_name=art
set -e
eval "$($HOME/conda/bin/conda shell.bash hook)"

conda deactivate
conda env remove --name $env_name
conda create -n $env_name python=3.9 -y
conda activate $env_name

conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install mamba -c conda-forge -y
mamba install -c conda-forge pytorch-lightning kornia ftfy rich pandas scipy ipython omegaconf regex ipdb imageio imageio-ffmpeg einops librosa ffmpeg -y

pip install torch_optimizer
mamba install -c conda-forge youtokentome -y
pip install rudalle==0.0.1rc4
pip install lpips

mkdir lib
git clone https://github.com/CompVis/taming-transformers  ../lib/taming-transformers
pip install -e ../lib/taming-transformers

git clone https://github.com/openai/CLIP.git ../lib/CLIP
pip install -e ../lib/CLIP

git clone https://github.com/shariqfarooq123/AdaBins.git ../lib/AdaBins
# pip install -e ../lib/AdaBins

git clone https://github.com/crowsonkb/guided-diffusion ../lib/guided-diffusion
pip install -e ../lib/guided-diffusion

mkdir weights -p; cd weights
curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
curl -OL https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
curl -OL https://the-eye.eu/public/AI/models/512x512_diffusion_unconditional_ImageNet/512x512_diffusion_uncond_finetune_008100.pt
curl -OL https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet.pth
curl -OL https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth
echo "FINISHED SETTING UP ENVIROMENT _art_"
