#!/bin/bash
cd ..
mkdir -p miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda3/miniconda.sh

bash miniconda3/miniconda.sh -p
rm -rf miniconda3/miniconda.sh
