export PATH="/usr/local/bin:$PATH"
MPIROOT=${myprefix}/4.0.5_gcc10.2.0
export PATH=$MPIROOT/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MPIROOT/lib
export MANPATH=$MANPATH:$MPIROOT/share/man
export CPATH=`xcrun --show-sdk-path`/usr/include
