#!/bin/bash
cd ..
conda init
. ~/.bashrc
conda env create --name SOM_env -f Linux_install/lin_tutosub.yml
. ~/.bashrc
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate SOM_env
jupyter notebook