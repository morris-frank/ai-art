#! /bin/sh
set -e
eval "$($HOME/conda/bin/conda shell.bash hook)"
conda activate art

if [ ! -d ../lib/taming-transformers ]; then
    git clone https://github.com/CompVis/taming-transformers  ../lib/taming-transformers
fi

mkdir weights; cd weights
curl -OL https://heibox.uni-heidelberg.de/f/867b05fc8c4841768640/?dl=1
curl -OL https://heibox.uni-heidelberg.de/f/274fb24ed38341bfa753/?dl=1
cd ..