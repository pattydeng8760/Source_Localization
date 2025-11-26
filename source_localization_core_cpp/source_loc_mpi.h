#IFNDEF SOURCE_LOC_MPI_H
#define SOURCE_LOC_MPI_H
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <mpi.h>
#include <omp.h>
#include <vector>
#include <complex>
#include <cmath>
#include <stdexcept>
#include <rarray>
#include "source_loc_single_obs.h"

namespace py = pybind11;

py::tuple compute_acoustic_surface_pressure_mpi(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> p_hat_in,       // (nodes, nfreq_all)
    py::array_t<double,             py::array::c_style | py::array::forcecast> zeta_in,          // (nodes, 3)
    py::array_t<double,             py::array::c_style | py::array::forcecast> normal_in,        // (nodes, 3)
    py::array_t<double,             py::array::c_style | py::array::forcecast> area_in,          // (nodes,)
    py::array_t<double,             py::array::c_style | py::array::forcecast> freq_all_in,      // (nfreq_all,)
    py::array_t<double,             py::array::c_style | py::array::forcecast> target_freq_in,   // (n_target_freq,)
    double speed_of_sound
);

#ENDIF // SOURCE_LOC_MPI_H