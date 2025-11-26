#!/usr/bin/env python3
import time
import math
import os
import numpy as np
import source_loc_cpp  # C++ MPI+OMP backend


def get_mpi_rank_size_from_env():
    """
    Get MPI rank/size from environment variables (OpenMPI style).
    Falls back to rank=0, size=1 if not running under mpirun.
    """
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    return rank, size


def build_test_case(nodes: int, nfreq_all: int, n_target_freq: int):
    """
    Build a synthetic but non-trivial test case for the C++ module.
    Geometry, normals, area, p_hat, and frequencies are all generated here.
    """

    rng = np.random.default_rng(12345)

    # Random geometry: points in a box around origin
    zeta = rng.uniform(low=-0.5, high=0.5, size=(nodes, 3)).astype(np.float64)

    # Random unit normals at each node
    raw_normals = rng.normal(size=(nodes, 3))
    normal = raw_normals / np.linalg.norm(raw_normals, axis=1)[:, None]

    # Uniform area (or you can randomize if you like)
    area = np.full(nodes, 1.0e-4, dtype=np.float64)

    # Frequency grid
    f_max = 5000.0  # Hz, arbitrary
    freq_all = np.linspace(0.0, f_max, nfreq_all, dtype=np.float64)

    # Random complex pressure spectrum (nodes, nfreq_all)
    p_hat = np.zeros((nodes, nfreq_all), dtype=np.complex128)
    for i in range(nodes):
        amp = 0.5 + 0.5 * rng.random()
        center = rng.uniform(0.0, nfreq_all - 1)
        width = rng.uniform(1.0, 0.3 * nfreq_all)
        envelope = np.exp(
            -0.5 * ((np.arange(nfreq_all) - center) / width) ** 2
        )
        phase = rng.uniform(0.0, 2.0 * math.pi, size=nfreq_all)
        p_hat[i, :] = amp * envelope * np.exp(1j * phase)

    # Target frequencies: pick a subset from freq_all
    idx = np.linspace(0, nfreq_all - 1, n_target_freq, dtype=int)
    target_frequencies = freq_all[idx].copy()

    c0 = 343.0  # speed of sound (m/s)

    return p_hat, zeta, normal, area, freq_all, target_frequencies, c0


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Pure C++ source_loc_cpp timing test (no mpi4py)"
    )
    parser.add_argument("--nodes", type=int, default=50000,
                        help="Number of surface nodes (default: 2000)")
    parser.add_argument("--nfreq", type=int, default=512,
                        help="Total number of frequencies (default: 512)")
    parser.add_argument("--ntarget", type=int, default=32,
                        help="Number of target frequencies (default: 32)")
    args = parser.parse_args()

    nodes = args.nodes
    nfreq_all = args.nfreq
    n_target_freq = args.ntarget

    rank, size = get_mpi_rank_size_from_env()

    if rank == 0:
        print(f"[INFO] MPI ranks (env) = {size}")
        print(f"[INFO] OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS', 'not set')}")
        print(f"[INFO] nodes={nodes}, nfreq_all={nfreq_all}, n_target_freq={n_target_freq}", flush=True)

    # Build test case (each rank builds the same inputs; C++ handles MPI distribution)
    p_hat, zeta, normal, area, freq_all, target_frequencies, c0 = \
        build_test_case(nodes, nfreq_all, n_target_freq)

    # Timing only on rank 0 (approximate wall-clock)
    # All ranks will execute the same Python code; MPI synchronization is
    # handled inside the C++ module.
    t0 = time.perf_counter()

    p_hat_s, target_indices = source_loc_cpp.compute_acoustic_surface_pressure_mpi(
        p_hat,
        zeta,
        normal,
        area,
        freq_all,
        target_frequencies,
        c0,
    )

    t1 = time.perf_counter()

    if rank == 0:
        elapsed = t1 - t0
        print(
            f"TIMING seconds={elapsed:.6f} ranks={size} threads={os.environ.get('OMP_NUM_THREADS', 'unknown')}",
            flush=True
        )

        # Light sanity check on shapes
        assert p_hat_s.shape == (n_target_freq, nodes)
        assert target_indices.shape == (n_target_freq,)
        print("[INFO] p_hat_s shape:", p_hat_s.shape)
        print("[INFO] target_indices shape:", target_indices.shape)


if __name__ == "__main__":
    main()