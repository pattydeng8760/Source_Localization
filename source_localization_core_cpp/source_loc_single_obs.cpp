/// @file source_loc_single_obs.cpp
/// @brief Compute surface integral for a single observer position.
/// @author Patrick G.C. Deng
/// @date November 25, 2025
/// @details This file implements the function to compute the surface integral for source localization at one observer position for one frequency
/// Calculates p̂_S(x) = (1/2π) ∫ [exp(-ikr)(ikr + 1) * (e_r · n_ξ) / r²] * p̂(ξ) dS(ξ) at a single observer position x 
/// Equivalent to p̂_S(x) = (1/2π) ∑ [exp(-ikr)(ikr + 1) * (e_r · n_ξ) / r²] * p̂(ξ) * ΔS(ξ) over all source elements

#include "source_loc_single_obs.h"

/// @param x_obs Observer position (3,)
/// @param zeta Source element node positions (nodes, 3) row-major
/// @param normal Source element normals (nodes, 3) row-major
/// @param area Source element areas (nodes,)
/// @param p_current Source element current pressure in frequency domain (nodes,)
/// @param nodes Number of source elements
/// @param k Wavenumber

std::complex<double> compute_surface_integral_single_obs(
    const rvector<double>& x_obs, const rarray<double,2>& zeta, const rarray<double,2>& normal, const rarray <double,1>& area, const rarray<std::complex<double>,1>& p_current, int nodes, double k){
    // Constants for intitialization
    double sum_re = 0.0;                    // Real part accumulator
    double sum_im = 0.0;                    // Imaginary part accumulator
    // Parallel loop over source elements, with reduction to accumulate results to sum_re and sum_im to avoid race conditions within each thread
    #pragma omp parallel for reduction(+:sum_re,sum_im)
    for (int src_idx = 0; src_idx < nodes; ++src_idx) {
        // Source position on surface
        double sx = zeta[src_idx][0];
        double sy = zeta[src_idx][1];
        double sz = zeta[src_idx][2];
        // Compute vector from source to observer as r_vec = observer - source
        double rx = x_obs[0] - sx;
        double ry = x_obs[1] - sy;
        double rz = x_obs[2] - sz;
        // Compute distance r = |r_vec|
        double r = std::sqrt(rx*rx + ry*ry + rz*rz);
        // skip singular point as the definition of the principal value integral
        if (r < 1e-12) {
            continue; 
        }
        // Computing the unit vector e_r = r_vec / r
        double ex = rx / r;
        double ey = ry / r;
        double ez = rz / r;
        // Compute the unit normal vector n_xi at the source element
        double er_n = ex*normal[src_idx][0] + ey*normal[src_idx][1] + ez*normal[src_idx][2];
        double kr = k * r;

        // exp(-i k r) = exp(0 - i*k*r)
        const std::complex<double> exp_neg_ikr = std::exp(std::complex<double>(0.0, -kr));
        // (i k r + 1)
        std::complex<double> ikr_plus(1.0, k*r);

        // Green's factor: exp(-i k r) * (i k r + 1) * (e_r·n_xi) / r^2
        std::complex<double> green_factor = exp_neg_ikr * ikr_plus * (er_n / (r*r));

        // Contribution from this source element
        std::complex<double> contrib = green_factor * p_current[src_idx] * area[src_idx];

        // Accumulate real and imaginary parts 
        sum_re += contrib.real();
        sum_im += contrib.imag();
    }

    std::complex<double> surface_integral(sum_re, sum_im);
    surface_integral /= (2.0 * M_PI);
    return surface_integral;
}