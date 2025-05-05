# chemqulacs-gpu

Quantum Chemistry Code with High-Performance Quantum Circuit Simulator

## Feature 
- NVIDIA cuStateVec backend
- Support distributed state-vector on mpi GPU cluster
  - Efficient communication by lazy qubit reordering
- Python - C++ interface by pybind

## Usage

### Dependencies
Expect these to be installed on modern supercomputers.

- [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html)
    - Validated: SingularityPRO version 3.7-10.el8
- OpenMPI
    - Validated: 4.1.4
- NVIDIA GPU
    - Validated: NVIDIA A100, V100

### Build

Execute [`build.sh`](../../build.sh) located under `chemqulacs-gpu`.

```bash
# PWD:chemqulacs-gpu
./build.sh
```
This will generate the chemqulacs-gpu Singularity container and compile the source code under `chemqulacs/cpp`. 
Please be patient as it takes about 20 minutes, depending on the environment.

### Run

There is a sample script in [`examples/08.runvqe-mpicpp/run.py`](./examples/08.runvqe-mpicpp/run.py).

#### Single-GPU execution

```bash
singularity exec --nv --bind .:/opt/chemqulacs-gpu ./singularity/chemqulacs.sif python examples/08.runvqe-mpicpp/run.py
```

#### Single-node and multi-gpu execution

```bash
singularity exec --nv --bind .:/opt/chemqulacs-gpu ./singularity/chemqulacs.sif mpirun -np 4 python examples/08.runvqe-mpicpp/run.py
```

This is the case with 4 GPUs in a node.

#### Multi-node and multi-gpu execution

```bash
mpirun -np 16 -npernode 8 singularity exec --nv --bind .:/opt/chemqulacs-gpu ./singularity/chemqulacs.sif python examples/08.runvqe-mpicpp/run.py
```

This is an example of using two compute nodes with 8 GPUs for a total of 16 GPUs.
We have confirmed that the system can run on up to 2048 GPUs.

## Publication

* Yusuke Teranishi, Shoma Hiraoka, Wataru Mizukami, Masao Okita, Fumihiko Ino, Lazy Qubit Reordering for Accelerating Parallel State-Vector-based Quantum Circuit Simulation, [arXiv:2410.04252](https://arxiv.org/abs/2410.04252) (2024)
* Yusuke Teranishi, Shoma Hiraoka, Wataru Mizukami, Hybrid Parallel Quantum Chemistry Simulation for Large-Scale GPU Clusters, [NVIDIA GTC25 P73003](https://www.nvidia.com/en-us/on-demand/session/gtc25-P73003/) (2025)

## License
MIT

## Support
The chemqulacs-gpu is supported by JST COI-NEXT ‘Quantum Software Research Hub’, GrantNumber JPMJPF2014.
