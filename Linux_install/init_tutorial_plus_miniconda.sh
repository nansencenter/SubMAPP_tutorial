#!/bin/bash
cd ..
mkdir -p miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda3/miniconda.sh

bash miniconda3/miniconda.sh -b -u -p miniconda3
rm -rf miniconda3/miniconda.sh
echo "export PATH=$PWD/miniconda3/bin:$PATH" >> ~/.bashrc
conda init
. ~/.bashrc
conda env create --name SOM_env -f Linux_install/lin_tutosub.yml
. ~/.bashrc
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate SOM_env
jupyter notebook
