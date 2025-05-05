/**
 * @file execute_manager.hpp
 * @brief Header file for ExecuteManager class
 * @author Yusuke Teranishi
 */
#pragma once

#include <cublas_v2.h>
#include <custatevec.h>
#include <mpi.h>

#include <dlfcn.h>
#include <cstdlib>
#include <vector>

/**
 * @brief Macro for handling CUDA errors
 */
#define HANDLE_CUDA(call)                                                                   \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaStatus != cudaSuccess) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: HANDLE_CUDA \"%s\" in line %d of file %s failed "               \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit(cudaStatus);                                                               \
        }                                                                                   \
    }

/**
 * @brief Macro for handling cuStateVec errors
 */
#define HANDLE_CUSV(call)                                                                         \
    {                                                                                             \
        custatevecStatus_t cudaStatus = call;                                                     \
        if (cudaStatus != CUSTATEVEC_STATUS_SUCCESS) {                                            \
            fprintf(stderr,                                                                       \
                    "ERROR: HANDLE_CUSV \"%s\" in line %d of file %s failed "                     \
                    "with "                                                                       \
                    "%s (%d).\n",                                                                 \
                    #call, __LINE__, __FILE__, custatevecGetErrorString(cudaStatus), cudaStatus); \
            exit(cudaStatus);                                                                     \
        }                                                                                         \
    }

/**
 * @brief Macro for handling cuBLAS errors
 */
#define HANDLE_CUBLAS(call)                                                                    \
    {                                                                                          \
        cublasStatus_t cudaStatus = call;                                                      \
        if (cudaStatus != CUBLAS_STATUS_SUCCESS) {                                             \
            fprintf(stderr,                                                                    \
                    "ERROR: HANDLE_CUBLAS \"%s\" in line %d of file %s failed "                \
                    "with "                                                                    \
                    "%s (%d).\n",                                                              \
                    #call, __LINE__, __FILE__, cublasGetStatusString(cudaStatus), cudaStatus); \
            exit(cudaStatus);                                                                  \
        }                                                                                      \
    }

/**
 * @brief Macro for handling MPI errors
 */
#define HANDLE_MPI(call)                                                              \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (mpi_status != MPI_SUCCESS) {                                              \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (mpi_error_string != NULL) {                                           \
                fprintf(stderr,                                                       \
                        "ERROR: HANDLE_MPI \"%s\" in line %d of file %s failed "      \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            } else {                                                                  \
                fprintf(stderr,                                                       \
                        "ERROR: HANDLE_MPI \"%s\" in line %d of file %s failed "      \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
            }                                                                         \
            exit(mpi_status);                                                         \
        }                                                                             \
    }

#define USE_EXTERNAL_MPI_COMMUNICATOR

extern void (*custatevecSetMpiCommunicator)(MPI_Comm);

/**
 * @brief Class for managing execution
 */
class ExecuteManager {
   private:
    int _mpi_rank;
    int _mpi_size;
    int _n_devices_per_node;
    int _device_id;

    cudaStream_t _cuda_stream;
    cudaEvent_t _cuda_event;
    custatevecHandle_t _cusv_handle;
    cublasHandle_t _cublas_handle;

    custatevecCommunicatorDescriptor_t _cusv_comm;

   public:
    /**
     * @brief Constructor
     */
    ExecuteManager() {
        HANDLE_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &_mpi_rank));
        HANDLE_MPI(MPI_Comm_size(MPI_COMM_WORLD, &_mpi_size));
        HANDLE_CUDA(cudaGetDeviceCount(&_n_devices_per_node));
        _device_id = _mpi_rank % _n_devices_per_node;
        cudaSetDevice(_device_id);

        HANDLE_CUDA(cudaStreamCreate(&_cuda_stream));
        HANDLE_CUDA(cudaEventCreateWithFlags(&_cuda_event, cudaEventInterprocess | cudaEventDisableTiming));
        HANDLE_CUSV(custatevecCreate(&_cusv_handle));
        HANDLE_CUSV(custatevecSetStream(_cusv_handle, _cuda_stream));
        HANDLE_CUBLAS(cublasCreate(&_cublas_handle));
        HANDLE_CUBLAS(cublasSetStream(_cublas_handle, _cuda_stream));
#ifdef USE_EXTERNAL_MPI_COMMUNICATOR
        HANDLE_CUSV(custatevecCommunicatorCreate(_cusv_handle, &_cusv_comm, CUSTATEVEC_COMMUNICATOR_TYPE_EXTERNAL,
                                                 "mpicomm.so"));
        void* lib = dlopen("mpicomm.so", RTLD_LAZY);
        if (lib == NULL) {
            printf("%s\n", dlerror());
            exit(1);
        }
        custatevecSetMpiCommunicator = (void (*)(MPI_Comm))dlsym(lib, "custatevecSetMpiCommunicator");
        dlclose(lib);
#else
        custatevecStatus_t cusv_status =
            custatevecCommunicatorCreate(_cusv_handle, &_cusv_comm, CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI, nullptr);
        if (cusv_status != CUSTATEVEC_STATUS_SUCCESS) {
            HANDLE_CUSV(custatevecCommunicatorCreate(_cusv_handle, &_cusv_comm, CUSTATEVEC_COMMUNICATOR_TYPE_OPENMPI,
                                                     "libmpi.so"));
        }
#endif
    }

    /**
     * @brief Destructor
     */
    ~ExecuteManager() {
        HANDLE_CUSV(custatevecCommunicatorDestroy(_cusv_handle, _cusv_comm));
        HANDLE_CUBLAS(cublasDestroy(_cublas_handle));
        HANDLE_CUSV(custatevecDestroy(_cusv_handle));
        HANDLE_CUDA(cudaEventDestroy(_cuda_event));
        HANDLE_CUDA(cudaStreamDestroy(_cuda_stream));
    }

    int getMpiRank() const { return _mpi_rank; }                   ///< Get MPI rank
    int getMpiSize() const { return _mpi_size; }                   ///< Get MPI size
    int getDevicesPerNode() const { return _n_devices_per_node; }  ///< Get the number of devices per node
    int getDeviceId() const { return _device_id; }                 ///< Get the device ID

    /**
     * @brief Get the available device memory size
     * @return Available device memory size
     */
    size_t getAvailableDeviceMemSize() {
        size_t free_size;
        size_t total_size;
        HANDLE_CUDA(cudaMemGetInfo(&free_size, &total_size));
        // printf("free=%zu,total=%zu\n", free_size, total_size);
        return free_size;
    }

    cudaStream_t getCudaStream() const { return _cuda_stream; }        ///< Get the CUDA stream
    cudaEvent_t getCudaEvent() const { return _cuda_event; }           ///< Get the CUDA event
    custatevecHandle_t getCusvHandle() const { return _cusv_handle; }  ///< Get the cuStateVec handle
    cublasHandle_t getCublasHandle() const { return _cublas_handle; }  ///< Get the cuBLAS handle

    custatevecCommunicatorDescriptor_t getCusvCommunicator() const {
        return _cusv_comm;
    }  ///< Get the cuStateVec communicator
};

extern ExecuteManager _ExecuteManager;
