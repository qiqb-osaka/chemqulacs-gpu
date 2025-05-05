#!/bin/bash -e

cd singularity
./build_sif.sh

cd ..
echo "Begin: Compile"
singularity exec --nv --no-home --bind chemqulacs/cpp:$PWD singularity/chemqulacs.sif /bin/bash -c "mkdir -p build && cd build && cmake -Dpybind11_DIR=\`python3 -m pybind11 --cmakedir\` .. && make"
echo "End: Compile"
