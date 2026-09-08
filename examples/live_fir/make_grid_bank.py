#!/usr/bin/env python
"""Write a small deterministic aligned-spin BBH template bank as HDF5.

Two banks are produced from one (mchirp, eta) grid:
  - a dense "fine" bank
  - a sparse "coarse" reference bank (every Nth point)

No stochastic placement -- this is a fast, reproducible fixture for the
FIR/ratio Live example, not a search bank.
"""
import argparse

import numpy as np
import h5py

from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--fine-out", default="bank_fine.hdf")
p.add_argument("--coarse-out", default="bank_coarse.hdf")
p.add_argument("--approximant", default="IMRPhenomD")
p.add_argument("--f-lower", type=float, default=20.0)
p.add_argument("--mchirp-min", type=float, default=6.0)
p.add_argument("--mchirp-max", type=float, default=12.0)
p.add_argument("--n-mchirp", type=int, default=30)
p.add_argument("--eta-min", type=float, default=0.18)
p.add_argument("--eta-max", type=float, default=0.25)
p.add_argument("--n-eta", type=int, default=4)
p.add_argument("--coarse-stride", type=int, default=6)
args = p.parse_args()

mc = np.linspace(args.mchirp_min, args.mchirp_max, args.n_mchirp)
eta = np.linspace(args.eta_min, args.eta_max, args.n_eta)
MC, ETA = np.meshgrid(mc, eta)
MC, ETA = MC.ravel(), ETA.ravel()
m1 = mass1_from_mchirp_eta(MC, ETA)
m2 = mass2_from_mchirp_eta(MC, ETA)
order = np.argsort(MC)
m1, m2 = m1[order], m2[order]
n = len(m1)


def write(path, sel):
    with h5py.File(path, "w") as f:
        f["mass1"] = m1[sel]
        f["mass2"] = m2[sel]
        f["spin1z"] = np.zeros(len(sel))
        f["spin2z"] = np.zeros(len(sel))
        f["f_lower"] = np.full(len(sel), args.f_lower)
        f["approximant"] = np.array([args.approximant.encode()] * len(sel))
        f.attrs["minimal_match"] = 0.97
    print("wrote %s (%d templates)" % (path, len(sel)))


write(args.fine_out, np.arange(n))
write(args.coarse_out, np.arange(0, n, args.coarse_stride))
