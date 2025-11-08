/// @file source_localization.cpp
/// @brief C++ implementation of the source localization on the basis of surface pressure "A quantity to identify turbulence related sound generation on
/// surfaces", Delfs, J. Sound Vib. 586, 2024 (https://doi.org/10.1016/j.jsv.2024.118490)
/// @author Patrick Deng
/// @date 2025-06-15

#include <rarray>                       // Rarray header for array operations      
#include <iostream>                     // Standard I/O header
#include <mpi.h>                        // MPI header to distribute the work amonst the processes
#include <omp.h>                        // OpenMP header to parallelize the work on each process
#include "ticktock.h"                   // Header for timing the execution of the code

void source_localization( )