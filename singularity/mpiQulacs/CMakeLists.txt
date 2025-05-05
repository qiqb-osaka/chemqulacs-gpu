cmake_minimum_required(VERSION 3.0)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)
if (MSVC)
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/wd 4819\"")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/wd 4244\"")
    if (USE_OMP)
        list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/openmp\"")
        list(APPEND CUDA_NVCC_FLAGS "--default-stream per-thread")
    endif()
endif()

file(GLOB GPUSIM_SRC
    "*.cu"
    "*.cuh"
    "*.h")

message("-- CUDA --")
message("Version: " ${CUDA_VERSION})
message("Library: " ${CUDA_CUDA_LIBRARY})
message("Runtime: " ${CUDA_CUDART_LIBRARY})

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  message("-DCMAKE_CUDA_ARCHITECTURES is not set, automatic architecture detection...")
  include(FindCUDA/select_compute_arch)
  CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
  string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
  string(REPLACE " " ";" INSTALLED_GPU_CCS_3 "${INSTALLED_GPU_CCS_2}")
  string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_3}")
  SET(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH_LIST})
  set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")
  set_property(GLOBAL PROPERTY CUDA_STANDARD 17) # with cmake 3.13 this is not enough, -std=c++17 is needed below
  message("setting CUDA_ARCHITECTURES global property to ${CUDA_ARCH_LIST}")
else()
  message("-DCMAKE_CUDA_ARCHITECTURES is set...")
  set(CUDA_ARCH_LIST "${CMAKE_CUDA_ARCHITECTURES}")
  message("setting CUDA_ARCH_LIST to ${CMAKE_CUDA_ARCHITECTURES}")
  set_property(GLOBAL PROPERTY CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  set_property(GLOBAL PROPERTY CUDA_STANDARD 17) # with cmake 3.13 this is not enough, -std=c++17 is needed below
  message("setting CUDA_ARCHITECTURES global property to ${CMAKE_CUDA_ARCHITECTURES}")
  set(ARCH_FLAGS "--generate-code=arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=[compute_${CMAKE_CUDA_ARCHITECTURES},sm_${CMAKE_CUDA_ARCHITECTURES}]")
endif()


#CUDA_GENERATED_OUTPUT_DIR(CMAKE_CURRENT_BINARY_DIR / build)
#CUDA_curand_LIBRARY
#cmake-- help - module FindCUDA

CUDA_COMPILE(gpusim_static ${GPUSIM_SRC})
CUDA_COMPILE(gpusim_shared ${GPUSIM_SRC})

CUDA_ADD_LIBRARY(gpusim_static ${GPUSIM_SRC} STATIC)
CUDA_ADD_LIBRARY(gpusim_shared ${GPUSIM_SRC} SHARED)

target_link_libraries(gpusim_static ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})
target_link_libraries(gpusim_shared ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})

message("Flag: " ${CUDA_NVCC_FLAGS})

set_target_properties(gpusim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
