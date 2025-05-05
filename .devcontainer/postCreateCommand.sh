#!/bin/bash

echo "module purge" >> ~/.bashrc
echo "module load nvhpc" >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc

# chemqulacsの依存ライブラリインストール
cd ~/workspace/chemqulacs-gpu
poetry install

# qulacsのインストール
source ./.venv/bin/activate
cd ~/workspace/qulacs
cp ~/workspace/chemqulacs-gpu/singularity/mpiQulacs/CMakeLists.txt ./src/gpusim/CMakeLists.txt
C_COMPILER=mpicc CXX_COMPILER=mpic++ USE_MPI=Yes USE_GPU=Yes pip install .
deactivate
