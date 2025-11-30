#!/usr/bin/env python3
import sys
import os
import math
import numpy as np
import time
import logging
from datetime import datetime

# Ensure source_localization_core is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
from source_localization_core.source_localization_utils import (
    compute_acoustic_surface_pressure_reference,
)
from source_localization_core import source_localization_core_cpp 
def get_mpi_rank_size_from_env():
    """Get MPI rank/size from environment variables (OpenMPI style)."""
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
    size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    return rank, size


def build_toy_problem(nodes: int, nfreq_all: int):
    """
    Build a simple, deterministic test case for comparing
    the Python reference and C++ implementations.
    """

    # Geometry: nodes along x-axis, y=z=0
    x_coords = np.linspace(0.0, 0.09, nodes)
    zeta = np.zeros((nodes, 3), dtype=float)
    zeta[:, 0] = x_coords  # x

    # Random unit normals at each node (fixed seed for reproducibility)
    rng = np.random.default_rng(12345)
    raw_normals = rng.normal(size=(nodes, 3))
    normal = raw_normals / np.linalg.norm(raw_normals, axis=1)[:, None]

    # Uniform area
    area = np.full(nodes, 1.0e-4, dtype=float)

    # Frequency grid
    freq_all = np.linspace(0.0, 2000.0, nfreq_all, dtype=float)

    # Deterministic complex p_hat: (nodes, nfreq_all)
    # p_hat[i, j] = exp(-i 2π f_j * (i * Δt)), Δt = 1e-4
    dt = 1.0e-4
    p_hat = np.zeros((nodes, nfreq_all), dtype=np.complex128)
    for i in range(nodes):
        for j in range(nfreq_all):
            f = freq_all[j]
            p_hat[i, j] = np.exp(-1j * 2.0 * math.pi * f * (i * dt))

    return zeta, normal, area, freq_all, p_hat


def main():
    rank, size = get_mpi_rank_size_from_env()
    
    # Keep the test modest so the reference is not insanely slow
    nodes = 5000
    nfreq_all = 8
    c0 = 343.0
    
    if rank == 0:
        date_str = datetime.now().strftime("%Y%m%d")
        log_path = os.path.join(os.getcwd(), f"log_compare_source_loc_{nodes}nodes_{date_str}.txt")
        sys.stdout = open(log_path, "w")
        sys.stdout.reconfigure(line_buffering=True)  # auto flush each line

    if rank == 0:
        print(f"MPI ranks (env) = {size}")
        print(f"nodes={nodes}, nfreq_all={nfreq_all}", flush=True)

    zeta, normal, area, freq_all, p_hat = build_toy_problem(nodes, nfreq_all)

    # One or two target frequencies; start with one for speed
    target_frequencies = [float(freq_all[2])]
    if rank == 0:
        print("target_frequencies:", target_frequencies)

    # --- C++ computation ---
    if rank == 0:
        print("\nCalling C++ compute_acoustic_surface_pressure_mpi...")
    t1 = time.time()
    p_hat_s_cpp, target_indices_cpp = source_localization_core_cpp.compute_acoustic_surface_pressure_mpi(
        p_hat,
        zeta,
        normal,
        area,
        freq_all,
        target_frequencies,
        c0,
    )
    t2 = time.time()
    if rank == 0:
        print("      p_hat_s_cpp shape:", p_hat_s_cpp.shape)
        print("      target_indices_cpp:", target_indices_cpp)
        print("C++ computation time: %.3f seconds" % (t2 - t1))

    # Python reference compute 
    if rank == 0:
        print("\nCalling Python reference compute_acoustic_surface_pressure_reference...")
    t1= time.time()
    p_hat_s_py, target_indices_py = compute_acoustic_surface_pressure_reference(
        p_hat,
        zeta,
        freq_all,
        normal,
        area,
        target_frequencies,
        c0,
    )
    t2 = time.time()
    if rank == 0:
        print("      p_hat_s_py shape:", p_hat_s_py.shape)
        print("      target_indices_py:", target_indices_py)
        print("Python reference computation time: %.3f seconds" % (t2 - t1))
        # --- Compare full fields ---
        if p_hat_s_py.shape != p_hat_s_cpp.shape:
            print("[ERROR] Shape mismatch between Python and C++ results!")
            print("  Python:", p_hat_s_py.shape)
            print("  C++   :", p_hat_s_cpp.shape)
            return

        diff = p_hat_s_cpp - p_hat_s_py
        max_abs_diff = np.max(np.abs(diff))
        mean_abs_diff = np.mean(np.abs(diff))

        print("\nComparison between C++ and Python reference:")
        print("      Max |Δp_s| :", max_abs_diff)
        print("      Mean |Δp_s|:", mean_abs_diff)

        # Sample a few nodes for detailed comparison
        obs_nodes_to_check = [0, 3, nodes // 2, nodes - 1]
        print("\nSample values at selected observer nodes:")
        for fi, f in enumerate(target_frequencies):
            for obs_node in obs_nodes_to_check:
                val_py = p_hat_s_py[fi, obs_node]
                val_cpp = p_hat_s_cpp[fi, obs_node]
                d = abs(val_cpp - val_py)
                print(
                    f"       node {obs_node:4d}: "
                    f"Python={val_py:.6e}, C++={val_cpp:.6e}, |Δ|={d:.3e}"
                )
        print("\nDone.")


if __name__ == "__main__":
    main()