#!/bin/bash -eu

singularity exec --nv --bind ../../:/opt/chemqulacs-gpu ../../singularity/chemqulacs.sif python vqe_h2o_sto3g.py
