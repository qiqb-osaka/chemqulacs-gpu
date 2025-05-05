#!/bin/bash -e

cp chemqulacs_cpp.pyi chemqulacs_cpp.pyx
CC=gcc poetry run python setup.py build_ext --inplace
rm -rf chemqulacs_cpp.pyx ./build chemqulacs_cpp.c  
