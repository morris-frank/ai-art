#! /bin/sh
set -e
eval "$($HOME/conda/bin/conda shell.bash hook)"
conda activate art

mamba install -c conda-forge youtokentome -y
pip install rudalle==0.0.1rc4