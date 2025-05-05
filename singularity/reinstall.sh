#!/bin/bash -e

singularity run -f --nv --no-home -w --bind ..:/opt/chemqulacs-gpu chemqulacs_sandbox
