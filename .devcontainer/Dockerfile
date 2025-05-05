FROM nvcr.io/nvidia/nvhpc:22.11-devel-cuda11.8-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# Create non-root user
RUN apt-get update && apt-get install -y sudo
ARG USERNAME
ARG UID
ARG GID
RUN groupadd --gid $GID $USERNAME \
	&& useradd --uid $UID --gid $GID -m $USERNAME -s /bin/bash \
	&& echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME \
	&& chmod 0440 /etc/sudoers.d/$USERNAME

# NVHPC
ENV MPI_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/comm_libs/openmpi4/openmpi-4.0.5
ENV PATH=$MPI_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$MPI_HOME/lib:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/math_libs/lib64:$LIBRARY_PATH
RUN cd /opt/nvidia/hpc_sdk/Linux_x86_64/22.11/compilers/lib/ && mv libgomp.so.1 libgomp.so.1.bk

# Standard compiler
ENV CC=gcc
ENV CXX=g++
ENV FC=gfortran
ENV F77=gfortran
ENV F90=gfortran
ENV CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/cuda
ENV OMPI_CC=$CC
ENV OMPI_CXX=$CXX
ENV OMPI_FC=$FC

RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
	build-essential \
	ca-certificates \
	less \
	curl \
	openssl \
	libssl-dev \
	libbz2-dev \
	libreadline-dev \
	libsqlite3-dev \
	libffi-dev \
	zlib1g-dev \
	liblzma-dev \
  libboost-dev \
	python3-dev \
	libncursesw5-dev \
	xz-utils \
	tk-dev \
	libxml2-dev \
	libxmlsec1-dev \
	libgdbm-dev \
	libgdbm-compat-dev \
	locales

# Setup locale
RUN sed -i -e 's/# \(ja_JP.UTF-8\)/\1/' /etc/locale.gen
RUN locale-gen
RUN update-locale LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8
ENV LANG=ja\_JP.UTF-8

# Install cuQuantum & cuTensor
ENV INSTALL_DIR=/opt
ENV CUQUANTUM_ROOT=${INSTALL_DIR}/cuquantum
ENV CUTENSOR_ROOT=${INSTALL_DIR}/cutensor
ENV CUQUANTUM_FILENAME=cuquantum-linux-x86_64-23.06.1.8_cuda11-archive
ENV CUTENSOR_FILENAME=libcutensor-linux-x86_64-1.7.0.1-archive
WORKDIR ${INSTALL_DIR}
RUN wget -nv https://developer.download.nvidia.com/compute/cuquantum/redist/cuquantum/linux-x86_64/${CUQUANTUM_FILENAME}.tar.xz
RUN wget -nv https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/${CUTENSOR_FILENAME}.tar.xz
RUN tar xf ${CUQUANTUM_FILENAME}.tar.xz
RUN tar xf ${CUTENSOR_FILENAME}.tar.xz
RUN mv ${CUQUANTUM_FILENAME} cuquantum
RUN mv ${CUTENSOR_FILENAME} cutensor
RUN rm ${CUQUANTUM_FILENAME}.tar.xz
RUN rm ${CUTENSOR_FILENAME}.tar.xz
ENV LD_LIBRARY_PATH=${CUQUANTUM_ROOT}/lib:${CUTENSOR_ROOT}/lib/11:${LD_LIBRARY_PATH}

# Install osu-benchmark
ENV OSU_FILENAME=osu-micro-benchmarks-7.3
RUN cd /opt; \
  wget -nv http://mvapich.cse.ohio-state.edu/download/mvapich/${OSU_FILENAME}.tar.gz; \
  tar --no-same-permissions --no-same-owner -xf ${OSU_FILENAME}.tar.gz; \
  rm ${OSU_FILENAME}.tar.gz; \
  cd ${OSU_FILENAME}; \
  ./configure CC=mpicc CXX=mpicxx \
	--enable-cuda \
	--with-cuda=$CUDA_HOME \
	--with-cuda-libpath=$CUDA_HOME/lib64/stubs; \
  make

# Change non-root user
USER ${USERNAME}

# Install pyenv
ENV PYTHON_VERSION=3.11.11
ENV PYENV_ROOT=/home/${USERNAME}/.pyenv
ENV PATH=${PYENV_ROOT}/bin:${PATH}
RUN git clone https://github.com/pyenv/pyenv.git ${PYENV_ROOT}
RUN eval "$(pyenv init -)";\
  pyenv install ${PYTHON_VERSION};\
  pyenv global ${PYTHON_VERSION};\
  pyenv rehash

# Install poetry
ENV POETRY_HOME=/home/${USERNAME}/.poetry
RUN eval "$(pyenv init -)";\
  curl -sSL https://install.python-poetry.org | POETRY_HOME=${POETRY_HOME} python3 -
ENV PATH=$POETRY_HOME/bin:$PATH

# clone qulacs 
WORKDIR /home/${USERNAME}/workspace
RUN git clone https://github.com/qulacs/qulacs.git
