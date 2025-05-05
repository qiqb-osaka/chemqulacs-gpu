#!/bin/bash

ATDYN=/path/to/genesis/bin/atdyn
mpirun -np 1 $ATDYN inp >& out
