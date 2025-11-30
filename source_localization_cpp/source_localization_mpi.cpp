/// @file source_localization_mpi.cpp
/// @brief Compute surface integral for multiple observer positions using MPI and OpenMP parallelization.
/// @author Patrick G.C. Deng
/// @date November 25, 2025
/// @details This file implements the function to compute the surface integral for source localization at multiple observer positions and frequencies using MPI for distributed computing and OpenMP for shared-memory parallelism.
/// Calculates p̂_S(x) = (1/2π) ∫ [exp(-ikr)(ikr + 1) * (e_r · n_ξ) / r²] * p̂(ξ) dS(ξ) for all observer positions parallelized with MPI
/// Equivalent to p̂_S(x) = (1/2π) ∑ [exp(-ikr)(ikr + 1) * (e_r · n_ξ) / r²] * p̂(ξ) * ΔS(ξ) 

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include "source_localization_worker.h"
#include "source_localization_mpi.h"
namespace py = pybind11;

// Track whether this module itself called MPI_Init
static bool g_mpi_initialized_here = false;

// Finalize MPI at program exit if we initialized it
static void finalize_mpi_at_exit()
{
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized && g_mpi_initialized_here) {
        MPI_Finalize();
    }
}

/// @param p_hat_in Surface pressure spectrum on all nodes and frequencies (nodes, nfreq_all)
/// @param zeta_in Node coordinates (nodes, 3)
/// @param normal_in Surface normals (nodes, 3)
/// @param area_in Surface element areas (nodes,)
/// @param freq_all_in Full FFT frequency vector (nfreq_all,)
/// @param target_freq_in Frequencies to extract from freq_all (nearest neighbor) (n_target_freq,)
/// @param speed_of_sound Speed of sound used to compute k = 2*pi*f/c

py::tuple compute_acoustic_surface_pressure_mpi(
    py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast> p_hat_in,       // (nodes, nfreq_all)
    py::array_t<double,             py::array::c_style | py::array::forcecast> zeta_in,          // (nodes, 3)
    py::array_t<double,             py::array::c_style | py::array::forcecast> normal_in,        // (nodes, 3)
    py::array_t<double,             py::array::c_style | py::array::forcecast> area_in,          // (nodes,)
    py::array_t<double,             py::array::c_style | py::array::forcecast> freq_all_in,      // (nfreq_all,)
    py::array_t<double,             py::array::c_style | py::array::forcecast> target_freq_in,   // (n_target_freq,)
    double speed_of_sound
)
{
    // --- MPI init / rank info ---
    int mpi_initialized = 0;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        MPI_Init(nullptr, nullptr);
        g_mpi_initialized_here = true;
        std::atexit(finalize_mpi_at_exit);
    }
    // Get MPI rank and size from the global communicator
    int world_size = 1;
    int world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // --- Extract buffers and shapes from py::array_t ---
    auto p_hat_buf       = p_hat_in.request();
    auto zeta_buf        = zeta_in.request();
    auto normal_buf      = normal_in.request();
    auto area_buf        = area_in.request();
    auto freq_all_buf    = freq_all_in.request();
    auto target_freq_buf = target_freq_in.request();

    if (p_hat_buf.ndim != 2) {
        throw std::runtime_error("p_hat must be 2D: (nodes, nfreq_all)");
    }
    if (zeta_buf.ndim != 2 || zeta_buf.shape[1] != 3) {
        throw std::runtime_error("zeta must be (nodes, 3)");
    }
    if (normal_buf.ndim != 2 || normal_buf.shape[1] != 3) {
        throw std::runtime_error("normal must be (nodes, 3)");
    }
    if (area_buf.ndim != 1) {
        throw std::runtime_error("area must be 1D: (nodes,)");
    }
    if (freq_all_buf.ndim != 1) {
        throw std::runtime_error("freq_all must be 1D");
    }
    if (target_freq_buf.ndim != 1) {
        throw std::runtime_error("target_frequencies must be 1D");
    }

    const int nodes      = static_cast<int>(p_hat_buf.shape[0]);
    const int nfreq_all  = static_cast<int>(p_hat_buf.shape[1]);
    const int nfreq_all_check = static_cast<int>(freq_all_buf.shape[0]);
    if (nfreq_all_check != nfreq_all) {
        throw std::runtime_error("freq_all length must match second dimension of p_hat");
    }
    if (zeta_buf.shape[0] != nodes || normal_buf.shape[0] != nodes || area_buf.shape[0] != nodes) {
        throw std::runtime_error("zeta, normal, area must all have same number of nodes as p_hat");
    }

    // Extract the number of target frequencies
    const int n_target_freq = static_cast<int>(target_freq_buf.shape[0]);
    // Pointers to input data
    auto* p_hat_ptr    = static_cast<std::complex<double>*>(p_hat_buf.ptr); // (nodes, nfreq_all)
    auto* zeta_ptr     = static_cast<double*>(zeta_buf.ptr);                // (nodes, 3)
    auto* normal_ptr   = static_cast<double*>(normal_buf.ptr);              // (nodes, 3)
    auto* area_ptr     = static_cast<double*>(area_buf.ptr);                // (nodes,)
    auto* freq_all_ptr = static_cast<double*>(freq_all_buf.ptr);            // (nfreq_all,)
    auto* target_ptr   = static_cast<double*>(target_freq_buf.ptr);         // (n_target_freq,)

    // --- 1) Map target frequencies to closest freq_all indices ---
    std::vector<int>    target_indices(n_target_freq);
    std::vector<double> target_values(n_target_freq);

    // Find closest frequency in freq_all for the input target frequency
    for (int i = 0; i < n_target_freq; ++i) {
        const double tf = target_ptr[i];
        double best_diff = std::abs(freq_all_ptr[0] - tf);
        int best_idx = 0;
        for (int j = 1; j < nfreq_all; ++j) {
            double diff = std::abs(freq_all_ptr[j] - tf);
            if (diff < best_diff) {
                best_diff = diff;
                best_idx  = j;
            }
        }
        const double closest_freq = freq_all_ptr[best_idx];
        target_indices[i] = best_idx;
        target_values[i]  = closest_freq;
    }

    // --- 2) Build p_hat_target: (n_target_freq, nodes) in freq-major layout ---
    std::vector<std::complex<double>> p_hat_target(
        static_cast<size_t>(n_target_freq) * nodes
    );

    for (int fi = 0; fi < n_target_freq; ++fi) {
        const int idx = target_indices[fi]; // index in [0, nfreq_all)
        for (int node = 0; node < nodes; ++node) {
            // p_hat_in layout: (nodes, nfreq_all)
            p_hat_target[static_cast<size_t>(fi)*nodes + node] =
                p_hat_ptr[node*nfreq_all + idx];
        }
    }

    // --- 3) Copy geometry/area into rarray containers for clean indexing ---
    rarray<double,2> zeta_arr(nodes, 3);
    rarray<double,2> normal_arr(nodes, 3);
    rarray<double,1> area_arr(nodes);

    for (int i = 0; i < nodes; ++i) {
        zeta_arr[i][0]   = zeta_ptr[i*3 + 0];
        zeta_arr[i][1]   = zeta_ptr[i*3 + 1];
        zeta_arr[i][2]   = zeta_ptr[i*3 + 2];

        normal_arr[i][0] = normal_ptr[i*3 + 0];
        normal_arr[i][1] = normal_ptr[i*3 + 1];
        normal_arr[i][2] = normal_ptr[i*3 + 2];

        area_arr[i]      = area_ptr[i];
    }

    // Temporary buffer for p_current for each frequency
    rarray<std::complex<double>,1> p_current(nodes);

    // --- 4) 1D block decomposition of observer nodes across MPI ranks ---
    int base = nodes / world_size;
    int rem  = nodes % world_size;

    auto node_start = [&](int r) {
        if (r < rem) {
            return r * (base + 1);
        } else {
            return rem * (base + 1) + (r - rem) * base;
        }
    };
    auto node_end = [&](int r) {
        if (r < rem) {
            return (r + 1) * (base + 1);
        } else {
            return rem * (base + 1) + (r - rem + 1) * base;
        }
    };

    const int obs_start   = node_start(world_rank);
    const int obs_end     = node_end(world_rank);
    const int nodes_local = obs_end - obs_start;

    // --- 5) Local buffer: (n_target_freq, nodes_local) ---
    std::vector<std::complex<double>> p_hat_s_local(
        static_cast<size_t>(n_target_freq) * nodes_local
    );

    // --- 6) Loop over target frequencies and local observer nodes ---
    for (int fi = 0; fi < n_target_freq; ++fi) {
        const double f = target_values[fi];
        const double k = (f == 0.0) ? 0.0 : (2.0 * M_PI * f / speed_of_sound);

        // Fill p_current (nodes) for this frequency from p_hat_target
        for (int node = 0; node < nodes; ++node) {
            p_current[node] = p_hat_target[static_cast<size_t>(fi)*nodes + node];
        }

        for (int local_idx = 0; local_idx < nodes_local; ++local_idx) {
            const int global_node = obs_start + local_idx;

            // Observer location x_obs = zeta_arr(global_node,:)
            rvector<double> x_obs(3);
            x_obs[0] = zeta_arr[global_node][0];
            x_obs[1] = zeta_arr[global_node][1];
            x_obs[2] = zeta_arr[global_node][2];

            std::complex<double> val;
            if (f == 0.0) {
                val = std::complex<double>(0.0, 0.0);
            } else {
                val = compute_surface_integral_single_obs(
                    x_obs,
                    zeta_arr,
                    normal_arr,
                    area_arr,
                    p_current,
                    nodes,
                    k
                );
            }

            p_hat_s_local[static_cast<size_t>(fi)*nodes_local + local_idx] = val;
        }
    }

    // --- 7) Allgatherv across ranks to assemble full p_hat_s on each process ---
    std::vector<std::complex<double>> p_hat_s_global;
    p_hat_s_global.resize(static_cast<size_t>(n_target_freq) * nodes);

    // recvcounts, displs in units of "nodes" for each rank
    std::vector<int> recvcounts(world_size);
    std::vector<int> displs(world_size);

    for (int r = 0; r < world_size; ++r) {
        int s = node_start(r);
        int e = node_end(r);
        recvcounts[r] = e - s;
        displs[r]     = s;
    }

    // For each frequency, gather nodes_local contributions along node dimension
    for (int fi = 0; fi < n_target_freq; ++fi) {
        const std::complex<double>* sendbuf =
            &p_hat_s_local[static_cast<size_t>(fi)*nodes_local];

        std::complex<double>* recvbuf =
            &p_hat_s_global[static_cast<size_t>(fi)*nodes];

        MPI_Allgatherv(
            sendbuf,
            nodes_local,
            MPI_DOUBLE_COMPLEX,
            recvbuf,
            recvcounts.data(),
            displs.data(),
            MPI_DOUBLE_COMPLEX,
            MPI_COMM_WORLD
        );
    }

    // --- 8) Wrap outputs as NumPy arrays ---
    // p_hat_s_out: (n_target_freq, nodes) freq-major
    py::array_t<std::complex<double>> p_hat_s_out({n_target_freq, nodes});
    auto p_hat_s_buf = p_hat_s_out.request();
    auto* p_hat_s_ptr = static_cast<std::complex<double>*>(p_hat_s_buf.ptr);

    std::copy(p_hat_s_global.begin(), p_hat_s_global.end(), p_hat_s_ptr);

    // target_indices as 1D array (int64)
    py::array_t<long long> idx_out(n_target_freq);
    auto idx_buf = idx_out.request();
    auto* idx_ptr = static_cast<long long*>(idx_buf.ptr);
    for (int i = 0; i < n_target_freq; ++i) {
        idx_ptr[i] = static_cast<long long>(target_indices[i]);
    }

    return py::make_tuple(p_hat_s_out, idx_out);
}


// pybind11 module definition
PYBIND11_MODULE(source_localization_core_cpp, m) {
    m.doc() = "C++/MPI/OpenMP backend for Delfs surface source localization";
    m.def(
        "compute_acoustic_surface_pressure_mpi",
        &compute_acoustic_surface_pressure_mpi,
        py::arg("p_hat"),
        py::arg("zeta"),
        py::arg("normal"),
        py::arg("area"),
        py::arg("freq_all"),
        py::arg("target_frequencies"),
        py::arg("speed_of_sound") = 343.0,
        R"pbdoc(
Compute acoustically active surface pressure p_hat_s for selected frequencies.

Parameters
----------
p_hat : complex ndarray, shape (nodes, nfreq_all)
    Surface pressure spectrum on all nodes and frequencies.
zeta : float64 ndarray, shape (nodes, 3)
    Node coordinates.
normal : float64 ndarray, shape (nodes, 3)
    Surface normals.
area : float64 ndarray, shape (nodes,)
    Surface element areas.
freq_all : float64 ndarray, shape (nfreq_all,)
    Full FFT frequency vector.
target_frequencies : float64 ndarray, shape (n_target_freq,)
    Frequencies to extract from freq_all (nearest neighbor).
speed_of_sound : float
    Speed of sound used to compute k = 2*pi*f/c.

Returns
-------
p_hat_s : complex ndarray, shape (n_target_freq, nodes)
    Acoustically active surface pressure for each target frequency.
target_indices : int64 ndarray, shape (n_target_freq,)
    Indices into freq_all corresponding to target_frequencies.
)pbdoc"
    );
}