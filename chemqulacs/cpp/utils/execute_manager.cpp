#include <utils/execute_manager.hpp>

#include <mpi.h>

void (*custatevecSetMpiCommunicator)(MPI_Comm);

ExecuteManager _ExecuteManager;
