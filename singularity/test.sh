#!/bin/bash

singularity exec --no-home --nv --bind ../:/opt/chemqulacs-gpu:rw ./chemqulacs.sif /bin/bash -c "cd /opt/chemqulacs-gpu && python -m pytest . -v --cache-clear"
