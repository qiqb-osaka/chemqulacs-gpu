#!/bin/bash -e

rm -rf chemqulacs_sandbox
singularity build -f --force --sandbox chemqulacs_sandbox chemqulacs.sif
