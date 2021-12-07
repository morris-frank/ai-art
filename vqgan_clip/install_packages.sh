#! /bin/sh
set -e
eval "$($HOME/conda/bin/conda shell.bash hook)"
conda activate art

pip install torch_optimizer

if [ ! -d ../lib/taming-transformers ]; then
    git clone https://github.com/CompVis/taming-transformers  ../lib/taming-transformers
    pip install -e ../lib/taming-transformers
fi

if [ ! -d ../lib/CLIP ]; then
    git clone https://github.com/openai/CLIP.git ../lib/CLIP
    pip install -e ../lib/CLIP
fi

mkdir weights -p; cd weights
curl -L -o vqgan_imagenet_f16_16384.yaml -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' #ImageNet 16384
curl -L -o vqgan_imagenet_f16_16384.ckpt -C - 'https://heibox.uni-heidelberg.de/d/a7530b09fed84f80a887/files/?p=%2Fckpts%2Flast.ckpt&dl=1' #ImageNet 16384
cd ..