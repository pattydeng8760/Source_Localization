import numpy as np
import source_loc_cpp  # built module
import math

def delfs_surface_integral_single_obs_python(x_obs, zeta, normal, area, p_current, k):
    """
    Pure Python / NumPy reference for a *single observer* and *single frequency*,
    matching the C++ compute_surface_integral_single_obs.
    """
    x_obs = np.asarray(x_obs, dtype=float)
    zeta = np.asarray(zeta, dtype=float)
    normal = np.asarray(normal, dtype=float)
    area = np.asarray(area, dtype=float)
    p_current = np.asarray(p_current, dtype=np.complex128)

    sum_val = 0.0 + 0.0j
    for i in range(zeta.shape[0]):
        sx, sy, sz = zeta[i]
        rx, ry, rz = x_obs - np.array([sx, sy, sz])
        r2 = rx*rx + ry*ry + rz*rz
        r = math.sqrt(r2)
        if r < 1e-12:
            continue
        ex, ey, ez = rx/r, ry/r, rz/r
        nx, ny, nz = normal[i]
        er_n = ex*nx + ey*ny + ez*nz
        kr = k*r
        exp_neg_ikr = np.exp(-1j*kr)
        ikr_plus_one = 1.0 + 1j*kr
        green = exp_neg_ikr * ikr_plus_one * (er_n / r2)
        sum_val += green * p_current[i] * area[i]
    return sum_val / (2.0*math.pi)


def main():
    # --- Toy problem setup ---
    nodes = 5000
    nfreq_all = 8

    # Simple geometry: nodes along x-axis, y=z=0
    x_coords = np.linspace(0.0, 0.09, nodes)
    zeta = np.zeros((nodes, 3), dtype=float)
    zeta[:, 0] = x_coords  # x
    # Normal pointing in +z just as a simple test
    rng = np.random.default_rng(12345)  # fixed seed for reproducibility
    raw_normals = rng.normal(size=(nodes, 3))  # Gaussian components
    normal = raw_normals / np.linalg.norm(raw_normals, axis=1)[:, None]
    # Uniform area
    area = np.full(nodes, 1.0e-4, dtype=float)

    # Frequency grid and test pressures
    freq_all = np.linspace(0.0, 2000.0, nfreq_all)
    # Random complex p_hat: shape (nodes, nfreq_all)
    # magnitude ~ O(1), smooth across frequency, random phase
    p_hat = np.zeros((nodes, nfreq_all), dtype=np.complex128)

    for i in range(nodes):
        # random reference amplitude per node
        amp = 0.5 + 0.5 * rng.random()
        # random smooth envelope in frequency using a Gaussian profile
        center = rng.uniform(0.0, nfreq_all - 1)
        width = rng.uniform(1.0, 0.3 * nfreq_all)
        envelope = np.exp(-0.5 * ((np.arange(nfreq_all) - center) / width) ** 2)

        # random phase for each frequency
        phase = rng.uniform(0, 2*np.pi, size=nfreq_all)

        p_hat[i, :] = amp * envelope * np.exp(1j * phase)
    for i in range(nodes):
        for j in range(nfreq_all):
            f = freq_all[j]
            # arbitrary toy pattern
            p_hat[i, j] = np.exp(-1j * 2*np.pi * f * (i*1e-4))

    # Choose a couple of target frequencies
    target_frequencies = np.array([freq_all[2], freq_all[5]], dtype=float)
    c0 = 343.0

    print("Calling C++ compute_acoustic_surface_pressure_mpi...")
    p_hat_s, target_indices = source_loc_cpp.compute_acoustic_surface_pressure_mpi(
        p_hat,
        zeta,
        normal,
        area,
        freq_all,
        target_frequencies,
        c0,
    )

    print("p_hat_s shape (C++):", p_hat_s.shape)  # (n_target_freq, nodes)
    print("target_indices:", target_indices)

    # --- Optional: compare 1 frequency / 1 observer with Python reference ---
    f0 = target_frequencies[0]
    idx0 = int(target_indices[0])
    k0 = 2.0 * math.pi * f0 / c0

    # Use full-surface p_current at that frequency
    p_current_0 = p_hat[:, idx0]

    # Pick an observer at node 3, say
    obs_node = 3
    x_obs = zeta[obs_node]

    p_s_python = delfs_surface_integral_single_obs_python(
        x_obs, zeta, normal, area, p_current_0, k0
    )

    p_s_cpp = p_hat_s[0, obs_node]

    print("\nComparison for frequency f = {:.2f} Hz at observer node {}:".format(f0, obs_node))
    print("Python reference:", p_s_python)
    print("C++ result       :", p_s_cpp)
    print("Absolute diff    :", abs(p_s_python - p_s_cpp))

if __name__ == "__main__":
    main()