# Copyright (C) 2026  Rahul Dhurkunde
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
# for more details.

"""Ratio-Filter Dechirping matched filter for PyCBC Live.

This is a drop-in alternative to :class:`pycbc.filter.matchedfilter.LiveBatchMatchedFilter`
that filters only a sparse *coarse* reference bank against the data and
reconstructs every dense *fine* template's SNR time series by convolving a
short FIR filter with the coarse reference's SNR (Nitz, Kacanja & Soni,
arXiv:2601.18835)::

    (s|h)(t) = F^-1[ h~(f) / h~_ref(f) ]  *  (s|h)_ref(t)

The FIR taps ``a_k`` are designed offline by ``pycbc_fir_bank`` such that
``FFT(a)(f) = (h~_fine(f) / h~_ref(f)) * sample_rate``.  With ``rho_ref`` the
*normalized* coarse SNR time series, the normalized fine SNR is then::

    rho_fine[n] = (sigma_ref / sigma_fine) / sample_rate
                  * sum_k rho_ref[n + k] a_k

i.e. a short cross-correlation of the coarse SNR with the tap sequence, scaled
by the fine/coarse sigma ratio.  This matches the normalization chain of the
offline ``MatchedFilterRatioControl`` / ``pycbc_inspiral_fir``.

This first implementation keeps the ratio stage in plain NumPy FFTs (no Cython
or GPU kernels); it targets correctness, not yet the paper's throughput.
"""

import logging

import numpy as np

from pycbc.filter.matchedfilter import matched_filter_core, correlate
from pycbc.types import zeros, complex64, TimeSeries, FrequencySeries
from pycbc.events import ranking

logger = logging.getLogger('pycbc.filter.matchedfilter_ratio_live')


def _prepare_fir_filters(taps, counts, nfft):
    """FFT a batch of FIR tap rows into search-ready frequency-domain filters.

    ``taps`` is ``(n_filters, k_max)`` (zero-padded); ``counts`` the valid tap
    count per row.  Each row is laid out circularly with its centre tap at
    index 0 (earlier taps wrap to the end), zero-padded to ``nfft``, forward
    FFT'd and conjugated so that ``ifft(fft(rho_ref_block) * filter)`` yields
    the cross-correlation ``sum_k rho_ref[n+k] a_k``.

    The reference SNR series is complex (analytic), so a full complex FFT is
    used, not rFFT.  Returns an ``(n_filters, nfft)`` complex128 array.
    """
    n_filters, k_max = taps.shape
    if k_max > nfft:
        raise ValueError("FIR tap count (%d) exceeds FFT block length (%d)"
                         % (k_max, nfft))
    buf = np.zeros((n_filters, nfft), dtype=np.float64)
    buf[:, :k_max] = taps
    # roll each row left by counts//2 so the centre tap lands at index 0
    rows = np.arange(n_filters)[:, None]
    cols = (np.arange(nfft)[None, :] + (counts // 2)[:, None]) % nfft
    buf = buf[rows, cols]
    return np.conj(np.fft.fft(buf, axis=1))


def _tap_autocorrelations(taps, counts):
    """One-sided autocorrelation ``A[j, tau] = sum_k a[j,k] a[j,k+tau]`` of each
    tap row, for ``tau`` in ``[0, k_max)``.  Used by the Parseval sigma path.
    """
    n_filters, k_max = taps.shape
    out = np.zeros((n_filters, k_max), dtype=np.float64)
    for j in range(n_filters):
        a = taps[j, :int(counts[j])].astype(np.float64)
        full = np.correlate(a, a, mode="full")
        out[j, :len(a)] = full[len(a) - 1:]
    return out


class RatioLiveBatchMatchedFilter(object):
    """Batched ratio/FIR matched filter over a rank's slice of coarse templates.

    Parameters
    ----------
    bank : pycbc.waveform.bank.LiveRatioFilterBank
        The hierarchical ratio bank.
    coarse_indices : iterable of int
        Coarse-bank indices this instance is responsible for.
    snr_threshold : float
        Record fine-template peaks with normalized SNR above this.
    chisq_bins : str
        Power-chisq bin specification (function of template params), as for
        ``LiveBatchMatchedFilter``.
    sg_chisq : pycbc.vetoes.sgchisq.SingleDetSGChisq
        Sine-Gaussian chisq calculator.
    fir_length : int
        FFT block length for the ratio-stage convolution.  Must exceed the
        valid analysis region plus the longest FIR filter; if too small it is
        grown automatically (with a warning).
    batch_size : int
        Number of fine filters to inverse-transform at once in the ratio stage.
    high_frequency_cutoff : float or None
        Upper cutoff for the reference matched filter (defaults to Nyquist).
    snr_abort_threshold, newsnr_threshold, max_triggers_in_batch : optional
        Same meaning as in ``LiveBatchMatchedFilter``.
    template_normalization_method : {'parseval', 'precalculated_sigma', 'mchirp'}
        How the fine template's SNR normalization (sigma) is obtained.
        ``parseval`` computes ``(h_fine|h_fine)`` under the *current* PSD from
        the reference-template autocorrelation and the FIR tap autocorrelation
        (arXiv:2601.18835 Eq. 6) -- correct as the live PSD drifts.  The other
        two scale the coarse template's live sigma by a fixed ratio recorded at
        bank-build time (``sigma_fine/sigma_ref``) or by ``(M_fine/M_ref)**5/6``.
    """

    def __init__(self, bank, coarse_indices, snr_threshold, chisq_bins,
                 sg_chisq, fir_length=32768, batch_size=64,
                 high_frequency_cutoff=None, snr_abort_threshold=None,
                 newsnr_threshold=None, max_triggers_in_batch=None,
                 template_normalization_method='precalculated_sigma'):
        self.bank = bank
        self.snr_threshold = snr_threshold
        self.snr_abort_threshold = snr_abort_threshold
        self.newsnr_threshold = newsnr_threshold
        self.max_triggers_in_batch = max_triggers_in_batch
        self.f_high = high_frequency_cutoff
        self.fir_length = int(fir_length)
        self.batch_size = int(batch_size)
        self.norm_method = template_normalization_method
        self.sample_rate = float(bank.sample_rate)

        from pycbc import vetoes
        self.power_chisq = vetoes.SingleDetPowerChisq(chisq_bins, None)
        self.sg_chisq = sg_chisq

        # Build a work item per coarse reference: the generated coarse
        # template, its associated fine templates, and their FFT'd FIR
        # filters.  Skip references that anchor no fine templates on this rank.
        self.groups = []
        for c in np.atleast_1d(coarse_indices):
            c = int(c)
            taps, counts, fine_idx = bank.get_firs(c)
            if len(fine_idx) == 0:
                continue
            ref = bank.get_coarse_template(c)
            k_max = int(counts.max())
            nfft = self._pick_nfft(k_max)
            filters_f = _prepare_fir_filters(taps, counts, nfft)
            group = {
                'coarse': c,
                'ref': ref,
                'delta_f': ref.delta_f,
                'fine_idx': np.asarray(fine_idx, dtype=np.int64),
                'counts': np.asarray(counts, dtype=np.int64),
                'k_max': k_max,
                'nfft': nfft,
                'filters_f': filters_f,
            }
            if self.norm_method == 'parseval':
                group['tap_autocorr'] = _tap_autocorrelations(taps, counts)
                group['_cref_cache'] = (None, None)
            else:
                group['snr_rescale'] = np.asarray(
                    bank.snr_rescale(fine_idx, self.norm_method),
                    dtype=np.float64)
            self.groups.append(group)

        # deterministic order, cheap to iterate
        self.groups.sort(key=lambda g: (g['delta_f'], g['coarse']))
        self.data = None
        logger.info('ratio MF: %d coarse references, %d fine templates',
                    len(self.groups),
                    sum(len(g['fine_idx']) for g in self.groups))

    def _pick_nfft(self, k_max):
        # Single block per increment: it must hold the valid analysis region
        # plus the FIR guard on both sides.  --fir-length is that block size;
        # grow it (to a power of two) if it is obviously too small for the
        # tap count.
        floor = 1 << int(np.ceil(np.log2(max(k_max * 4, 8192))))
        return int(max(self.fir_length, floor))

    # -- public API mirrors LiveBatchMatchedFilter -----------------------

    def set_data(self, data):
        self.data = data

    def process_data(self, data_reader):
        """Filter the current increment of ``data_reader`` and return a
        results dict (same schema as ``LiveBatchMatchedFilter``)."""
        self.set_data(data_reader)
        return self.process_all()

    def process_all(self):
        results = []
        veto_info = []
        for group in self.groups:
            res, veto = self._process_group(group)
            if res is False:
                return False
            if res is None:
                continue
            results.append(res)
            veto_info += veto

        if not results:
            return self._empty_result()

        result = self._combine(results)

        if self.max_triggers_in_batch and len(result['snr']):
            keep = result['snr'].argsort()[::-1][:self.max_triggers_in_batch]
            for key in result:
                result[key] = result[key][keep]
            veto_info = [veto_info[i] for i in keep]

        return self._process_vetoes(result, veto_info)

    # -- internals ------------------------------------------------------

    def _empty_result(self):
        keys = ['snr', 'coa_phase', 'end_time', 'template_id', 'sigmasq',
                'chisq', 'chisq_dof', 'sg_chisq']
        out = {k: np.array([], dtype=np.float32) for k in keys}
        out['snr'] = np.array([], dtype=np.float32)
        out['template_id'] = np.array([], dtype=np.uint64)
        out['chisq_dof'] = np.array([], dtype=np.uint32)
        for key in self._param_keys():
            out[key] = np.array([])
        return out

    def _param_keys(self):
        return list(self.bank.table.dtype.names)

    def _combine(self, results):
        result = {}
        for key in results[0]:
            result[key] = np.concatenate([r[key] for r in results])
        return result

    def _reference_snr(self, group):
        """Normalized coarse SNR time series (numpy complex128) for the
        current data increment, plus (sigma_ref, psd, stilde, valid slice)."""
        stilde = self.data.overwhitened_data(group['delta_f'])
        psd = stilde.psd
        ref = group['ref']
        h_norm = ref.sigmasq(psd)
        q, _, norm = matched_filter_core(
            ref, stilde, psd=None, h_norm=h_norm,
            low_frequency_cutoff=ref.f_lower,
            high_frequency_cutoff=self.f_high)
        rho_ref = q.numpy() * norm  # normalized complex SNR, length N

        n = len(rho_ref)
        valid_end = int(n - self.data.trim_padding)
        valid_start = int(valid_end
                          - self.data.blocksize * self.data.sample_rate)
        return rho_ref, np.sqrt(h_norm), psd, stilde, valid_start, valid_end

    def _group_sigma_fine(self, group, sigma_ref, psd):
        """Per-fine-template sigma under the current PSD via Parseval's theorem
        (arXiv:2601.18835 Eq. 6)::

            sigma_fine^2 = sum_tau C_ref(tau) * A_aa(tau) / sample_rate^2

        with ``C_ref`` the reference template's PSD-weighted autocorrelation
        (calibrated so ``C_ref(0) == sigma_ref^2``) and ``A_aa`` the FIR tap
        autocorrelation.  ``C_ref`` is cached per PSD object.
        """
        ref = group['ref']
        cached_id, cref = group['_cref_cache']
        if cached_id != id(psd):
            h = ref.numpy()
            p = psd.numpy()
            kmin = int(np.ceil(ref.f_lower / ref.delta_f))
            kmax = len(h) if self.f_high is None else \
                min(len(h), int(self.f_high / ref.delta_f) + 1)
            pw = np.zeros(len(h))
            good = slice(kmin, kmax)
            with np.errstate(divide='ignore', invalid='ignore'):
                pw[good] = np.abs(h[good]) ** 2 / p[good]
            pw = np.nan_to_num(pw)
            n = (len(h) - 1) * 2
            cref = np.fft.irfft(pw, n=n).real
            if cref[0] != 0:
                cref = cref * (sigma_ref ** 2 / cref[0])
            group['_cref_cache'] = (id(psd), cref)

        aa = group['tap_autocorr']            # (nfilt, k_max)
        kmax_t = aa.shape[1]
        w = np.empty(kmax_t)
        w[0] = cref[0]
        w[1:] = 2.0 * cref[1:kmax_t]
        s2 = (aa @ w) / (self.sample_rate ** 2)
        return np.sqrt(np.clip(s2, 1e-300, None))

    def _process_group(self, group):
        rho_ref, sigma_ref, psd, stilde, valid_start, valid_end = \
            self._reference_snr(group)

        guard = int(group['k_max'])  # generous wraparound guard
        vlen = valid_end - valid_start
        nfft = group['nfft']
        if nfft < vlen + 2 * guard:
            raise ValueError(
                "ratio MF: --fir-length (%d) too small for the %d-sample valid "
                "region + %d-tap guard; use at least %d" %
                (nfft, vlen, guard,
                 1 << int(np.ceil(np.log2(vlen + 2 * guard)))))
        w0 = valid_start - guard
        if w0 < 0:
            raise ValueError(
                "ratio MF: analysis region starts at %d, less than the %d-tap "
                "guard from the buffer start; increase --max-length" %
                (valid_start, guard))
        if (w0 + nfft) <= len(rho_ref):
            window = rho_ref[w0:w0 + nfft]
        else:
            window = np.zeros(nfft, dtype=np.complex128)
            window[:len(rho_ref) - w0] = rho_ref[w0:]
        win_f = np.fft.fft(window)

        # correlation output index range within the block that maps to
        # [valid_start, valid_end)
        off = valid_start - w0

        fine_idx = group['fine_idx']
        filters_f = group['filters_f']
        nfilt = len(fine_idx)

        # sigma of each fine template under the current PSD
        if self.norm_method == 'parseval':
            sigma_fine = self._group_sigma_fine(group, sigma_ref, psd)
        else:
            sigma_fine = sigma_ref * group['snr_rescale']

        peak_snr = np.zeros(nfilt, dtype=np.complex128)
        peak_pos = np.zeros(nfilt, dtype=np.int64)

        # rho_fine[n] = (sigma_ref/sigma_fine) / sample_rate * sum_k rho_ref[n+k] a_k
        # (the FIR taps carry a factor of sample_rate from their design:
        # FFT(a)(f) = ratio(f) * sample_rate).
        scale = sigma_ref / (sigma_fine * self.sample_rate)

        for s in range(0, nfilt, self.batch_size):
            e = min(s + self.batch_size, nfilt)
            corr = np.fft.ifft(win_f[None, :] * filters_f[s:e], axis=1)
            seg = corr[:, off:off + vlen]
            mag = seg.real * seg.real + seg.imag * seg.imag
            amax = mag.argmax(axis=1)
            rows = np.arange(e - s)
            peak_pos[s:e] = amax
            peak_snr[s:e] = seg[rows, amax]

        snr_val = peak_snr * scale
        snr_abs = np.abs(snr_val)

        keep = snr_abs >= self.snr_threshold
        if self.snr_abort_threshold is not None and \
                np.any(snr_abs[keep] > self.snr_abort_threshold):
            logger.info("ratio MF: implausibly loud SNR, abandoning chunk")
            return False, []

        if not np.any(keep):
            return None, None

        kidx = np.nonzero(keep)[0]
        ref = group['ref']
        start_time = self.data.start_time

        result = {}
        for key in self._param_keys():
            result[key] = self.bank.table[key][fine_idx[kidx]]
        result['snr'] = snr_abs[kidx].astype(np.float32)
        result['coa_phase'] = np.angle(snr_val[kidx]).astype(np.float32)
        result['end_time'] = (start_time
                              + peak_pos[kidx].astype(np.float64)
                              / self.data.sample_rate)
        result['template_id'] = fine_idx[kidx].astype(np.uint64)
        result['sigmasq'] = (sigma_fine[kidx] ** 2).astype(np.float32)
        if hasattr(ref, 'time_offset'):
            result['time_offset'] = np.full(len(kidx), ref.time_offset)

        veto_info = []
        for j, fi in zip(kidx, fine_idx[kidx]):
            veto_info.append((int(fi), int(peak_pos[j] + valid_start),
                              stilde, psd))
        return result, veto_info

    def _process_vetoes(self, results, veto_info):
        """Signal-consistency tests.

        The ratio engine only yields SNR, so chisq needs the fine template's
        own filter.  There are few triggers per increment in Live, so we
        reconstruct each surviving fine template (``ratio_filter * coarse``)
        and run the standard power- and SG-chisq against it -- exactly as
        ``LiveBatchMatchedFilter._process_vetoes`` does with a directly
        generated template.
        """
        n = len(veto_info)
        chisq = np.zeros(n, dtype=np.float32)
        dof = np.zeros(n, dtype=np.uint32)
        sg_chisq = np.zeros(n, dtype=np.float32)

        for i, (fi, l, stilde, psd) in enumerate(veto_info):
            htilde = self.bank.reconstruct_template(fi, delta_f=stilde.delta_f)
            nh = min(len(htilde), len(stilde))
            h_norm = htilde.sigmasq(psd)
            q, cout, norm = matched_filter_core(
                htilde[:nh], stilde[:nh], psd=None, h_norm=h_norm,
                low_frequency_cutoff=htilde.f_lower,
                high_frequency_cutoff=self.f_high)
            snrv = np.array([q.numpy()[l]])
            c, d = self.power_chisq.values(cout, snrv, norm, psd, [l], htilde)
            chisq[i] = c[0] / d[0]
            dof[i] = d[0]
            sgv = self.sg_chisq.values(stilde, htilde, psd, snrv, norm,
                                       c, d, [l])
            if sgv is not None:
                sg_chisq[i] = sgv[0]

        results['chisq'] = chisq
        results['chisq_dof'] = dof
        results['sg_chisq'] = sg_chisq

        if self.newsnr_threshold and n:
            newsnr = ranking.newsnr(results['snr'], chisq)
            keep = np.nonzero(newsnr >= self.newsnr_threshold)[0]
            for key in list(results):
                results[key] = results[key][keep]

        return results
