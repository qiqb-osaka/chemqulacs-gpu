CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
project(chemqulacs_cpp CXX)

set(PROJECT_VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -O3)

find_package(CUDA REQUIRED)
message("-- CUDA --")
message("CUDA_VERSION: " ${CUDA_VERSION})
message("CUDA_HOME: " ${CUDA_TOOLKIT_ROOT_DIR})

find_package(MPI REQUIRED)
message("-- MPI --")
message("MPI_CXX_VERSION: " ${MPI_CXX_VERSION})
message("MPI_CXX_COMPILER: " ${MPI_CXX_COMPILER})

find_package(OpenMP REQUIRED)

set(CUQUANTUM_ROOT $ENV{CUQUANTUM_ROOT})
message("-- cuQuantum --")
message("CUQUANTUM_ROOT: " ${CUQUANTUM_ROOT})

add_library(mpicomm SHARED mpi/mpicomm.cpp)
target_include_directories(mpicomm
  PRIVATE ${CUQUANTUM_ROOT}/include ${CUDA_INCLUDE_DIRS}
)
target_link_directories(mpicomm PRIVATE ${CUQUANTUM_ROOT}/lib)
target_link_libraries(mpicomm PRIVATE MPI::MPI_CXX custatevec ${CUDA_LIBRARIES})
set_target_properties(mpicomm PROPERTIES PREFIX "")

set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(target_directories
  "ansatz"
  "circuit"
  "expectation"
  "pauli"
  "pybind"
  "state"
  "utils"
)
list(TRANSFORM target_directories APPEND "/*.cpp" OUTPUT_VARIABLE target_sources_pattern)
file(GLOB target_sources RELATIVE ${ROOT_DIR} ${target_sources_pattern})

find_package(pybind11 REQUIRED)

pybind11_add_module(chemqulacs_cpp ${target_sources})
target_include_directories(chemqulacs_cpp
  PRIVATE ${ROOT_DIR} ${CUQUANTUM_ROOT}/include ${CUDA_INCLUDE_DIRS}
)
target_link_directories(chemqulacs_cpp PRIVATE ${CUQUANTUM_ROOT}/lib)
target_link_libraries(chemqulacs_cpp PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX pybind11::module custatevec cublas ${CUDA_LIBRARIES})
