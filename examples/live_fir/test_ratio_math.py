#!/usr/bin/env python
"""Validate the ratio-filter SNR reconstruction math and normalization scale.

Works entirely at the FIR bank's design resolution (delta_f), independent of
pycbc_live's adaptive buffer sizing.  For a sample of fine templates:
  * generate the true fine template and its coarse reference at delta_f
  * noiseless data = fine template injected at a known time / amplitude
  * SNR by a direct matched filter of the fine template          (reference)
  * SNR by the ratio method: matched-filter the coarse reference, cross-
    correlate rho_ref with the fine template's FIR taps, rescale by the
    fine/coarse sigma ratio
Reports fir/direct peak-SNR ratio (want ~1.0) and time error.

Usage: test_ratio_math.py FIR_BANK.hdf [--n 15] [--inj-snr 25] [--delta-f 0.5]
"""
import argparse

import numpy as np

import pycbc.psd
from pycbc.filter.matchedfilter import matched_filter_core, sigmasq
from pycbc.types import FrequencySeries, zeros
from pycbc.waveform import get_waveform_filter
from pycbc.filter.matchedfilter_ratio_live import _prepare_fir_filters
from pycbc import DYN_RANGE_FAC
import h5py

pa = argparse.ArgumentParser()
pa.add_argument("firbank")
pa.add_argument("--n", type=int, default=15)
pa.add_argument("--inj-snr", type=float, default=25.0)
pa.add_argument("--sample-rate", type=int, default=2048)
pa.add_argument("--delta-f", type=float, default=None,
                help="default: the bank's own design delta_f")
pa.add_argument("--psd-model", default="aLIGOZeroDetHighPower")
pa.add_argument("--approximant", default="IMRPhenomD")
pa.add_argument("--seed", type=int, default=3)
args = pa.parse_args()

SR = args.sample_rate
APX = args.approximant

f = h5py.File(args.firbank, "r")
fir = f["fir_data"]
DF = args.delta_f or float(fir.attrs.get("delta_f", 0.5))
flow = float(fir.attrs.get("f_low", 15.0))
flen = int(SR / 2 / DF) + 1
N = (flen - 1) * 2

tbl = {k: f[k][:] for k in ("mass1", "mass2", "spin1z", "spin2z")}
cb = {k: fir["coarse_bank_params"][k][:] for k in ("mass1", "mass2", "spin1z", "spin2z")}
gk = [k for k in fir if k.isdigit()]
# fine index -> (coarse key, local, taps, count, match)
fmap = {}
for k in gk:
    g = fir[k]
    for j, fi in enumerate(g["fine_bank_index"][:]):
        fmap[int(fi)] = (k, j, g["taps"][j], int(g["actual_tap_count"][j]),
                         float(g["filter_match"][j]))

# Work in double precision throughout: the raw aLIGO PSD is ~1e-46 strain^2/Hz,
# which underflows float32.  (pycbc_live's engine sees DYN_RANGE-scaled,
# single-precision data instead - a different, self-consistent regime.)
psd = pycbc.psd.from_string(args.psd_model, flen, DF, max(flow - 2.0, 1.0))


def gen(row):
    h = get_waveform_filter(zeros(flen, dtype=np.complex128), row,
                            approximant=APX, f_lower=flow, delta_f=DF,
                            delta_t=1.0 / SR, distance=1.0 / DYN_RANGE_FAC)
    return FrequencySeries(np.asarray(h.data, np.complex128), delta_f=DF)


rng = np.random.default_rng(args.seed)
pool = np.array([fi for fi, v in fmap.items() if v[4] > 0.99])
sel = rng.choice(pool, size=min(args.n, len(pool)), replace=False)

print("%6s %5s %9s %9s %8s %8s %7s" %
      ("fine", "cN", "rho_dir", "rho_fir", "fir/dir", "dt_ms", "fit_m"))
ratios = []
for fi in sel:
    ckey, local, taps_row, K, fitm = fmap[int(fi)]
    c = int(ckey)
    href = gen(dict(mass1=cb["mass1"][c], mass2=cb["mass2"][c],
                    spin1z=cb["spin1z"][c], spin2z=cb["spin2z"][c]))
    hfin = gen(dict(mass1=tbl["mass1"][fi], mass2=tbl["mass2"][fi],
                    spin1z=tbl["spin1z"][fi], spin2z=tbl["spin2z"][fi]))

    sig_f = np.sqrt(sigmasq(hfin, psd, low_frequency_cutoff=flow))
    if not np.isfinite(sig_f) or sig_f == 0:
        print("%6d  bad sigma_fine=%s" % (fi, sig_f))
        continue

    inj = N // 2
    ph = np.exp(-2j * np.pi * np.arange(flen) * DF * (inj / SR))
    s_raw = hfin.numpy() * ph * (args.inj_snr / sig_f)
    ow = np.zeros(flen, np.complex128)
    pv = psd.numpy()
    m = pv > 0
    ow[m] = s_raw[m] / pv[m]
    stilde = FrequencySeries(ow, delta_f=DF)

    q, _, nd = matched_filter_core(
        hfin, stilde, psd=None,
        h_norm=sigmasq(hfin, psd, low_frequency_cutoff=flow),
        low_frequency_cutoff=flow)
    rho_d = np.abs(q.numpy() * nd)
    id_d = rho_d.argmax()

    hnr = float(sigmasq(href, psd, low_frequency_cutoff=flow))
    qr, _, nr = matched_filter_core(href, stilde, psd=None, h_norm=hnr,
                                    low_frequency_cutoff=flow)
    rho_ref = qr.numpy() * nr   # normalized coarse SNR time series

    nfft = 1 << int(np.ceil(np.log2(N)))
    filt = _prepare_fir_filters(taps_row[None, :], np.array([K]), nfft)
    win = np.zeros(nfft, np.complex128)
    win[:N] = rho_ref
    corr = np.fft.ifft(np.fft.fft(win)[None] * filt, axis=1)[0]
    # rho_fine = corr / (rescale * sample_rate),  rescale = sigma_fine/sigma_ref
    rescale = float(fir[ckey]["sigmas"][local][1] / fir[ckey]["sigmas"][local][0])
    rho_fir = np.abs(corr / (rescale * SR))
    id_f = rho_fir[:N].argmax()

    ratio = rho_fir[id_f] / rho_d[id_d]
    ratios.append(ratio)
    print("%6d %5d %9.3f %9.3f %8.3f %8.2f %7.4f" %
          (fi, c, rho_d[id_d], rho_fir[id_f], ratio,
           (id_f - id_d) / SR * 1e3, fitm))

if ratios:
    r = np.array(ratios)
    print("\nfir/dir: mean %.4f  std %.4f  min %.4f  max %.4f  (n=%d)"
          % (r.mean(), r.std(), r.min(), r.max(), len(r)))
