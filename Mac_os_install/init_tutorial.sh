#!/bin/bash
cd ..
conda init
. ~/.bashrc
conda env create --name SOM_env -f Mac_os_install/lin_tutosub.yml
. ~/.bashrc
source "$(conda info --base)"/etc/profile.d/conda.sh
conda activate SOM_env
jupyter notebook