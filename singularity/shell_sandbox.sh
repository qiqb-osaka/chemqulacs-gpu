#!/bin/bash -e

singularity shell --nv --no-home -w --bind ..:/mnt chemqulacs_sandbox

# fakeroot
# singularity shell -f --nv --no-home -w --bind ..:/mnt chemqulacs_sandbox
