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
mamba install -c conda-forge pytorch-lightning kornia ftfy rich pandas scipy ipython omegaconf regex ipdb imageio imageio-ffmpeg einops -y

mkdir lib
echo "FINISHED SETTING UP ENVIROMENT _art_"
