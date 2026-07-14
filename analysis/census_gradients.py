#!/usr/bin/env python3
"""Full 168-config rerun storing per-observable gradient blocks.

Enables: (1) Ay-gain census at experimental precision ratios eps=5-10%,
(2) sigma_R gain census, (3) elastic-only multi-energy variants,
all as offline linear algebra afterwards.

Blocks per config (as stored by build_gradient_vector):
  G[:, :35]   dcs   d log sigma / d log p          (eps = 1)
  G[:, 35:70] Ay    (p/delta_Ay) dAy/dp            (delta_Ay = 0.03)
  G[:, 70]    sigma_R log-deriv
  G[:, 71]    sigma_T log-deriv (neutrons only)
"""
import sys, os
import numpy as np
from multiprocessing import Pool

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(REPO, 'src'))
sys.path.insert(0, os.path.join(REPO, 'analysis'))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'census_gradients.npz')

from deff_scan_extended import compute_fisher_extended, get_kd02_params_13

NUCLEI = [('12C',12,6), ('16O',16,8), ('27Al',27,13), ('28Si',28,14),
          ('40Ca',40,20), ('48Ti',48,22), ('56Fe',56,26), ('58Ni',58,28),
          ('90Zr',90,40), ('120Sn',120,50), ('197Au',197,79), ('208Pb',208,82)]
ENERGIES = [10, 20, 30, 50, 100, 150, 200]
THETA = np.linspace(5.0, 175.0, 35)

def one(args):
    proj, name, A, Z, E = args
    params = get_kd02_params_13(proj, A, Z, E)
    F, G, obs0, ndata = compute_fisher_extended(proj, A, Z, E, THETA, params)
    return (proj, name, A, Z, E, G)

if __name__ == '__main__':
    jobs = [(p, n, A, Z, E) for p in ('n','p') for (n,A,Z) in NUCLEI for E in ENERGIES]
    print(f"{len(jobs)} configs", flush=True)
    with Pool() as pool:
        out = []
        for i, r in enumerate(pool.imap_unordered(one, jobs)):
            out.append(r)
            if (i+1) % 12 == 0:
                print(f"  {i+1}/{len(jobs)} done", flush=True)
    keys = [f"{r[0]}_{r[1]}_{r[4]}" for r in out]
    np.savez_compressed(OUT,
        keys=np.array(keys),
        meta=np.array([(r[0], r[1], r[2], r[3], r[4]) for r in out], dtype=object),
        **{f"G_{k}": r[5] for k, r in zip(keys, out)})
    print(f"saved {OUT}", flush=True)
