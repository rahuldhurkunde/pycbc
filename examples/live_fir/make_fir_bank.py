#!/usr/bin/env python
"""Minimal, fast FIR/ratio bank builder for testing the Live engine.

A stripped-down stand-in for ``pycbc_fir_bank``: nearest coarse reference by
(tau0, tau3), a single PSD-weighted ridge least-squares FIR fit at a fixed tap
count (no escalation, no peak suppression), and the same output HDF layout the
Live ``LiveRatioFilterBank`` reads.  Fast enough to iterate; not tuned for a
production search.

Usage:
    make_fir_bank.py --coarse-bank C.hdf --fine-bank F.hdf --output-file O.hdf
        [--sample-rate 2048] [--f-low 15] [--delta-f 0.5] [--n-taps 251]
        [--decimation 4] [--ridge 1e-3] [--ratio-cap 500] [--psd-model ...]
"""
import argparse
import logging
import multiprocessing as mp

import numpy as np
import h5py
import scipy.linalg

import pycbc.psd
from pycbc.filter import match, sigma
from pycbc.pnutils import mass1_mass2_to_tau0_tau3
from pycbc.types import FrequencySeries, TimeSeries, zeros
from pycbc.waveform import get_waveform_filter
from pycbc import DYN_RANGE_FAC

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def gen(row, approx, flen, df, sr, flow):
    h = get_waveform_filter(zeros(flen, dtype=np.complex64), row,
                            approximant=approx, f_lower=flow, delta_f=df,
                            delta_t=1.0 / sr, distance=1.0 / DYN_RANGE_FAC)
    return FrequencySeries(np.asarray(h.data, dtype=np.complex128), delta_f=df)


def design_ridge_fir(ratio, band_mask, weights, K, dec, ridge):
    """PSD-weighted ridge least-squares FIR fit of a frequency-domain ratio.

    Mirrors pycbc_fir_bank.design_optimal_real_filter's 'ridge' path.
    ``ratio`` already carries the * sample_rate scaling (H_target_raw).
    """
    H = (ratio * 1.0)[::dec]
    m = band_mask[::dec]
    w = weights[::dec]
    n_nyq = len(H)
    n_full = 2 * (n_nyq - 1)
    fg = np.arange(n_nyq) / n_full

    vi = np.nonzero(m)[0]
    fv = fg[vi]
    b = H[vi]
    k_idx = np.arange(-(K - 1) // 2, (K - 1) // 2 + 1)
    A = np.exp(-1j * 2 * np.pi * np.outer(fv, k_idx))
    wv = w[vi]
    A *= wv[:, None]
    b = b * wv

    A_real = np.vstack([A.real, A.imag])
    b_real = np.concatenate([b.real, b.imag])
    reg = np.eye(K) * np.sqrt(ridge)
    A_final = np.vstack([A_real, reg])
    b_final = np.concatenate([b_real, np.zeros(K)])
    taps, *_ = scipy.linalg.lstsq(A_final, b_final, lapack_driver='gelsd')
    return taps


def fir_to_fd(taps, K, flen, df, sr):
    N = (flen - 1) * 2
    ts = np.zeros(N)
    start = K // 2
    ts[:K - start] = taps[start:]
    ts[-start:] = taps[:start]
    fs = TimeSeries(ts, delta_t=1.0 / sr).to_frequencyseries(delta_f=df)
    if len(fs) > flen:
        fs = FrequencySeries(fs.numpy()[:flen], delta_f=df)
    elif len(fs) < flen:
        fs = FrequencySeries(np.r_[fs.numpy(), np.zeros(flen - len(fs))], delta_f=df)
    return fs


_W = {}


def _init_worker(G):
    _W.clear()
    _W.update(G)
    _W['coarse_cache'] = {}
    logging.getLogger().setLevel(logging.WARNING)


def _row(**kw):
    return dict(kw)


def _coarse(c):
    cache = _W['coarse_cache']
    if c not in cache:
        cb, cf = _W['cb'], _W['c_flow']
        cache[c] = gen(_row(mass1=cb['mass1'][c], mass2=cb['mass2'][c],
                            spin1z=cb['spin1z'][c], spin2z=cb['spin2z'][c]),
                       _W['approx'], _W['flen'], _W['df'], _W['SR'], float(cf[c]))
    return cache[c]


def _fit(href, hfin_np):
    flen, band, SR, K = _W['flen'], _W['band'], _W['SR'], _W['K']
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.nan_to_num(hfin_np / href.numpy())
    w = np.zeros(flen)
    np.divide(np.abs(href.numpy()) ** 2, _W['psd_np'], out=w, where=band)
    w = np.nan_to_num(w)
    bmask = band & (np.abs(ratio) <= _W['ratio_cap'])
    if not bmask.any() or w[bmask].max() == 0:
        return None
    w = w / w[bmask].max()
    taps = design_ridge_fir(ratio * SR, bmask, w, K, _W['decimation'], _W['ridge'])
    Hfir = fir_to_fd(taps, K, flen, _W['df'], SR)
    hrec = FrequencySeries((href.numpy() * Hfir.numpy()).astype(np.complex128),
                           delta_f=_W['df'])
    return taps, hrec


def _worker(fi):
    fb, df, flow, K = _W['fb'], _W['df'], _W['flow'], _W['K']
    psd = _W['psd']
    hfin = gen(_row(mass1=fb['mass1'][fi], mass2=fb['mass2'][fi],
                    spin1z=fb['spin1z'][fi], spin2z=fb['spin2z'][fi]),
               _W['approx'], _W['flen'], df, _W['SR'], flow)
    hfin_c = FrequencySeries(hfin.numpy().astype(np.complex128), delta_f=df)
    best = None
    for c in _W['cand'][fi]:
        c = int(c)
        href = _coarse(c)
        r = _fit(href, hfin.numpy())
        if r is None:
            continue
        taps, hrec = r
        m, _ = match(hfin_c, hrec, psd=psd, low_frequency_cutoff=flow)
        if best is None or m > best[0]:
            cm, _ = match(hfin_c,
                          FrequencySeries(href.numpy().astype(np.complex128), delta_f=df),
                          psd=psd, low_frequency_cutoff=flow)
            best = (m, c, taps, hrec, cm)
            if m >= 0.9999:
                break
    if best is None:
        return (fi, None)
    m, c, taps, hrec, cm = best
    href = _coarse(c)
    s_ref = sigma(href, psd=psd, low_frequency_cutoff=flow)
    s_rec = sigma(hrec, psd=psd, low_frequency_cutoff=flow)
    s_tgt = sigma(hfin_c, psd=psd, low_frequency_cutoff=flow)
    return (fi, c, taps, float(m), float(cm),
            (float(s_ref), float(s_rec), float(s_tgt)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--coarse-bank", required=True)
    p.add_argument("--fine-bank", required=True)
    p.add_argument("--output-file", required=True)
    p.add_argument("--sample-rate", type=int, default=2048)
    p.add_argument("--f-low", type=float, default=15.0)
    p.add_argument("--delta-f", type=float, default=0.5)
    p.add_argument("--n-taps", type=int, default=251)
    p.add_argument("--decimation", type=int, default=4)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--ratio-cap", type=float, default=500.0)
    p.add_argument("--search-depth", type=int, default=4,
                   help="number of nearest coarse refs (by tau0,tau3) to try; "
                        "the one giving the best reconstruction match wins")
    p.add_argument("--n-processes", "-n", type=int, default=1)
    p.add_argument("--approximant", default="IMRPhenomXAS")
    p.add_argument("--psd-model", default="aLIGOZeroDetHighPower")
    args = p.parse_args()

    SR, df, flow = args.sample_rate, args.delta_f, args.f_low
    flen = int(SR / 2 / df) + 1
    K = args.n_taps | 1  # force odd
    psd = pycbc.psd.from_string(args.psd_model, flen, df, max(flow - 2, 1))
    psd_np = psd.numpy()
    freqs = np.arange(flen) * df
    band = (freqs >= flow) & (freqs < SR / 2)

    with h5py.File(args.coarse_bank, "r") as f:
        cb = {k: f[k][:] for k in ('mass1', 'mass2', 'spin1z', 'spin2z')}
        c_flow = f['f_lower'][:] if 'f_lower' in f else np.full(len(cb['mass1']), flow)
    with h5py.File(args.fine_bank, "r") as f:
        fb = {k: f[k][:] for k in f.keys()}
        f_attrs = dict(f.attrs)
    n_fine = len(fb['mass1'])
    n_coarse = len(cb['mass1'])
    logging.info("fine=%d coarse=%d flen=%d K=%d", n_fine, n_coarse, flen, K)

    # candidate coarse refs by (tau0, tau3) proximity, final choice by match
    ct0, ct3 = mass1_mass2_to_tau0_tau3(cb['mass1'], cb['mass2'], flow)
    ft0, ft3 = mass1_mass2_to_tau0_tau3(fb['mass1'], fb['mass2'], flow)
    s0, s3 = ct0.std() or 1.0, ct3.std() or 1.0
    cpts = np.column_stack([ct0 / s0, ct3 / s3])
    ndepth = min(args.search_depth, n_coarse)
    cand = np.array([
        np.argsort(((cpts - [ft0[i] / s0, ft3[i] / s3]) ** 2).sum(1))[:ndepth]
        for i in range(n_fine)])

    G = dict(cb=cb, c_flow=c_flow, fb=fb, cand=cand, psd=psd, psd_np=psd_np,
             band=band, flen=flen, df=df, SR=SR, flow=flow, K=K,
             approx=args.approximant, decimation=args.decimation,
             ridge=args.ridge, ratio_cap=args.ratio_cap)

    groups = {}  # c -> [(fine_i, taps, K, match, coarse_match, (s_ref,s_rec,s_tgt))]
    nfail = 0
    done = 0
    if args.n_processes > 1:
        pool = mp.Pool(args.n_processes, initializer=_init_worker, initargs=(G,))
        it = pool.imap_unordered(_worker, range(n_fine))
    else:
        _init_worker(G)
        it = map(_worker, range(n_fine))
    for res in it:
        done += 1
        if res[1] is None:
            nfail += 1
        else:
            fi, c, taps, m, cm, sig = res
            groups.setdefault(c, []).append(
                (fi, taps.astype(np.float32), K, m, cm, sig))
        if done % 25 == 0:
            logging.info("  %d/%d done (%d failed)", done, n_fine, nfail)
    if args.n_processes > 1:
        pool.close(); pool.join()
    if nfail:
        logging.warning("%d/%d fine templates got no usable FIR", nfail, n_fine)

    with h5py.File(args.output_file, "w") as o:
        for k, v in fb.items():
            o[k] = v
        for k, v in f_attrs.items():
            o.attrs[k] = v
        fg = o.create_group("fir_data")
        fg.attrs['sample_rate'] = SR
        fg.attrs['n_taps'] = K
        fg.attrs['min_match'] = 0.99
        fg.attrs['f_low'] = flow
        fg.attrs['delta_f'] = df
        cp = fg.create_group("coarse_bank_params")
        for k in ('mass1', 'mass2', 'spin1z', 'spin2z'):
            cp[k] = cb[k]
        cp['f_lower'] = c_flow
        cp['approximant'] = np.array([args.approximant.encode()] * n_coarse)
        cp.attrs['parameters'] = ['mass1', 'mass2', 'spin1z', 'spin2z',
                                  'f_lower', 'approximant']
        for c, rows in groups.items():
            g = fg.create_group(str(c))
            kmax = max(len(r[1]) for r in rows)
            taps = np.zeros((len(rows), kmax), dtype=np.float32)
            for j, r in enumerate(rows):
                taps[j, :len(r[1])] = r[1]
            g['taps'] = taps
            g['actual_tap_count'] = np.array([r[2] for r in rows], dtype=np.int32)
            g['fine_bank_index'] = np.array([r[0] for r in rows], dtype=np.int32)
            g['filter_match'] = np.array([r[3] for r in rows], dtype=np.float32)
            g['coarse_match'] = np.array([r[4] for r in rows], dtype=np.float32)
            g['sigmas'] = np.array([r[5] for r in rows], dtype=np.float32)

    allm = np.array([r[3] for rs in groups.values() for r in rs])
    logging.info("DONE %s: %d firs, match min %.4f median %.5f frac>0.99 %.2f",
                 args.output_file, len(allm), allm.min(), np.median(allm),
                 (allm > 0.99).mean())


if __name__ == "__main__":
    main()
