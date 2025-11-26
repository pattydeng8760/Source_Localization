#ifndef SOURCE_LOC_SINGLE_OBS_H
#define SOURCE_LOC_SINGLE_OBS_H

#include <omp.h>
#include <complex>
#include <cmath>
#include <rarray>
#include <algorithm>

std::complex<double> compute_surface_integral_single_obs(
    const rvector<double>& x_obs, const rarray<double,2>& zeta, const rarray<double,2>& normal, const rarray <double,1>& area, const rarray<std::complex<double>,1>& p_current, int nodes, double k);

#endif // SOURCE_LOC_SINGLE_OBS_H