#!/usr/bin/env python
"""Compare triggers between a standard pycbc_live run and a --ratio-bank-file
run over the same data/bank.

Matches triggers by detector + template_hash + end_time (<= tol) and reports:
  * trigger-count recovery
  * std of the SNR difference vs the mismatch bound sqrt(2*(1-min_match))
  * time / phase bias
  * agreement of the foreground coinc IFAR list

Usage: compare_live_triggers.py STD_DIR FIR_DIR [--fir-bank fir.hdf] [--tol-samples 1]
"""
import argparse
import glob
import os

import numpy as np
import h5py

pa = argparse.ArgumentParser()
pa.add_argument("std_dir")
pa.add_argument("fir_dir")
pa.add_argument("--fir-bank", default=None)
pa.add_argument("--sample-rate", type=int, default=2048)
pa.add_argument("--tol-samples", type=float, default=1.0)
pa.add_argument("--margin", type=float, default=4.0)
args = pa.parse_args()


def load_triggers(d):
    """Return {ifo: dict of arrays} pooled over all chunk files in d."""
    out = {}
    files = sorted(glob.glob(os.path.join(d, "*.hdf")))
    for fn in files:
        try:
            f = h5py.File(fn, "r")
        except OSError:
            continue
        with f:
            for ifo in f.keys():
                g = f[ifo]
                if not isinstance(g, h5py.Group) or "snr" not in g:
                    continue
                rec = out.setdefault(ifo, {})
                for key in ("snr", "end_time", "chisq", "template_hash",
                            "coa_phase", "template_id", "sigmasq"):
                    if key in g:
                        rec.setdefault(key, []).append(g[key][:])
    for ifo in out:
        for key in list(out[ifo]):
            out[ifo][key] = np.concatenate(out[ifo][key])
    return out, files


std, sfiles = load_triggers(args.std_dir)
fir, ffiles = load_triggers(args.fir_dir)
print("standard: %d chunk files, ifos %s" % (len(sfiles), sorted(std)))
print("fir     : %d chunk files, ifos %s" % (len(ffiles), sorted(fir)))
for ifo in sorted(std):
    print("  %s standard triggers: %d" % (ifo, len(std[ifo].get("snr", []))))
for ifo in sorted(fir):
    print("  %s fir      triggers: %d" % (ifo, len(fir[ifo].get("snr", []))))

tol = args.tol_samples / args.sample_rate
key = "template_hash"

all_d = []
for ifo in sorted(set(std) & set(fir)):
    s, r = std[ifo], fir[ifo]
    if "snr" not in s or "snr" not in r or key not in s or key not in r:
        print("  %s: missing snr/%s, skipping" % (ifo, key))
        continue
    matched = 0
    for i in range(len(r["snr"])):
        cand = np.nonzero((s[key] == r[key][i]) &
                          (np.abs(s["end_time"] - r["end_time"][i]) <= tol))[0]
        if len(cand):
            j = cand[np.argmin(np.abs(s["end_time"][cand] - r["end_time"][i]))]
            rchi = r["chisq"][i] if "chisq" in r else np.nan
            schi = s["chisq"][j] if "chisq" in s else np.nan
            rsig = r["sigmasq"][i] if "sigmasq" in r else np.nan
            ssig = s["sigmasq"][j] if "sigmasq" in s else np.nan
            all_d.append((ifo, r["snr"][i] - s["snr"][j],
                          r["end_time"][i] - s["end_time"][j],
                          r["snr"][i], s["snr"][j], rchi, schi, rsig, ssig))
            matched += 1
    print("  %s: matched %d / %d fir triggers to standard (%d standard)"
          % (ifo, matched, len(r["snr"]), len(s["snr"])))

if all_d:
    dsnr = np.array([x[1] for x in all_d])
    dt = np.array([x[2] for x in all_d])
    rchi = np.array([x[5] for x in all_d])
    schi = np.array([x[6] for x in all_d])
    print("\nSNR diff (fir - standard): mean %.4f  std %.4f  max|.| %.4f"
          % (dsnr.mean(), dsnr.std(), np.abs(dsnr).max()))
    print("time diff: mean %.2e s  max|.| %.2e s (%.2f samples)"
          % (dt.mean(), np.abs(dt).max(), np.abs(dt).max() * args.sample_rate))
    ok = np.isfinite(rchi) & np.isfinite(schi) & (schi > 0)
    if ok.any():
        dchi = (rchi[ok] - schi[ok]) / schi[ok]
        print("reduced-chisq frac diff (fir-std)/std: mean %.3f  std %.3f  "
              "max|.| %.3f" % (dchi.mean(), dchi.std(), np.abs(dchi).max()))
    rsig = np.array([x[7] for x in all_d])
    ssig = np.array([x[8] for x in all_d])
    oks = np.isfinite(rsig) & np.isfinite(ssig) & (ssig > 0)
    if oks.any():
        drs = np.sqrt(rsig[oks] / ssig[oks])
        print("sigma_fine ratio (fir/std): mean %.3f  std %.3f  "
              "min %.3f  max %.3f" % (drs.mean(), drs.std(), drs.min(), drs.max()))

    verdict_fail = False
    if args.fir_bank:
        with h5py.File(args.fir_bank, "r") as f:
            mm = float(f["fir_data"].attrs.get("min_match", 0.99))
        bound = np.sqrt(2 * (1 - mm))
        print("mismatch bound sqrt(2*(1-%.4f)) = %.4f ; measured/bound = %.2fx"
              % (mm, bound, dsnr.std() / bound))
        verdict_fail = dsnr.std() > args.margin * bound
        print("VERDICT:", "FAIL" if verdict_fail else "PASS",
              "(allowed %.1fx)" % args.margin)
    if np.abs(dt).max() * args.sample_rate > 2.0:
        print("VERDICT: FAIL (time offsets exceed 2 samples)")
        verdict_fail = True
else:
    verdict_fail = True
    print("\nno triggers matched between the two runs")


def coincs(d):
    out = []
    for fn in sorted(glob.glob(os.path.join(d, "*.hdf"))):
        try:
            f = h5py.File(fn, "r")
        except OSError:
            continue
        with f:
            if "foreground" in f and "ifar" in f["foreground"]:
                out.append((os.path.basename(fn),
                            float(np.atleast_1d(f["foreground/ifar"][()])[0])))
    return out


sc, fc = coincs(args.std_dir), coincs(args.fir_dir)
print("\nforeground coincs: standard %d, fir %d" % (len(sc), len(fc)))
for name, ifar in sc:
    print("  std %s ifar=%.4g" % (name, ifar))
for name, ifar in fc:
    print("  fir %s ifar=%.4g" % (name, ifar))

raise SystemExit(1 if verdict_fail else 0)
