"""Multi-reference ratio (multiband) filtering for PyCBC Live.

A drop-in replacement for LiveBatchMatchedFilter when the bank is a FIR
(ratio) bank written by pycbc_fir_bank. Each coarse reference template is
filtered once per stride at the full sample rate; every fine template is
filtered only below the split frequency, and rebuilt above it by applying
its FIR filter to the reference SNR, in the blocks that pass the low-band
gate. The filtering itself is RatioMatchedFilterControl, shared with
pycbc_inspiral_fir_multi, so live and offline give the same SNRs.
"""
import logging
import numpy

from pycbc.events import ranking
from pycbc.filter.matched_ratio import RatioMatchedFilterControl

logger = logging.getLogger('pycbc.filter.live_ratio')


class _Batch(object):
    """Fine templates of one coarse reference filtered together."""
    def __init__(self, bank, engine, coarse_index, indices):
        taps, counts, self.fine_ids = bank.get_firs(coarse_index,
                                                    indices=indices)
        self.coarse_index = coarse_index
        self.counts = counts
        self.templates = bank.get_fine_templates(coarse_index,
                                                 indices=indices)
        self.templates2 = numpy.empty((len(self.templates),
                                       len(self.templates[0])),
                                      dtype=numpy.complex64)
        for j, t in enumerate(self.templates):
            self.templates2[j, :] = t.data[:]
        self.filters_f, _ = engine.prepare_filters(taps, counts)
        self.scale_low = bank.snr_rescale(self.fine_ids, 'mchirp')
        self.scale_high = bank.snr_rescale(self.fine_ids,
                                           'precalculated_sigma')
        # The high band filters are prenormalized, as offline
        self.filters_f /= self.scale_high[:, None]
        # Low-band template norms depend on the PSD; cached per PSD object
        self.hnorms_psd = None
        self.hnorms_low = None


class LiveRatioBatchMatchedFilter(object):
    """Calculate SNR and signal consistency tests with the ratio filter for
    one MPI worker's share of a FIR bank.

    Parameters
    ----------
    bank: pycbc.waveform.bank.RatioFilterBank
        The FIR bank, generated with the stride segment's delta_f.
    param_table: pycbc.io.WaveformArray
        Template parameters of the fine bank (as in the live FilterBank), in
        file order; indexed by fine template id for the trigger output.
    worker: int
        This worker's index, from 0 to n_workers - 1.
    n_workers: int
        Number of filtering workers. Each takes a contiguous share of every
        coarse reference's fine templates.
    snr_threshold: float
        Minimum SNR to record a trigger.
    chisq_bins: str
        Power chisq bins, computed over the band below the split.
    multiband_frequency: float
        Split frequency of the FIR bank, in Hz.
    """
    def __init__(self, bank, param_table, worker, n_workers, snr_threshold,
                 chisq_bins, multiband_frequency, fir_length=4096,
                 batch_size=250, gate_safety_sigma=6.0,
                 gate_extra_margin=0.05, band_ratio_threshold=1.0,
                 snr_abort_threshold=None, newsnr_threshold=None,
                 max_triggers_in_batch=None):
        from pycbc import vetoes

        self.bank = bank
        self.param_table = param_table
        self.delta_f = bank.delta_f
        self.snr_threshold = snr_threshold
        self.snr_abort_threshold = snr_abort_threshold
        self.newsnr_threshold = newsnr_threshold
        self.max_triggers_in_batch = max_triggers_in_batch
        self.band_ratio_threshold = band_ratio_threshold
        self.power_chisq = vetoes.SingleDetPowerChisq(chisq_bins, None)

        self.engine = RatioMatchedFilterControl(
            snr_threshold=snr_threshold,
            delta_f=self.delta_f,
            fir_fft_length=fir_length,
            batch_size=batch_size,
            multiband_frequency=multiband_frequency,
            gate_safety_sigma=gate_safety_sigma,
            gate_extra_margin=gate_extra_margin,
        )

        # This worker's contiguous share of each coarse group, in batches
        self.batches = []
        self.coarse = {}
        for c in bank.coarse_indices:
            n = len(bank.get_fine_indices(c))
            edges = numpy.linspace(0, n, n_workers + 1).round().astype(int)
            start, stop = edges[worker], edges[worker + 1]
            if stop <= start:
                continue
            self.coarse[c] = bank.get_coarse_template(c)
            for s in range(start, stop, batch_size):
                self.batches.append(
                    _Batch(bank, self.engine, c,
                           slice(s, min(s + batch_size, stop))))
        ntmp = sum(len(b.fine_ids) for b in self.batches)
        logger.info('Ratio filter worker %d/%d: %d fine templates, '
                    '%d coarse references, %d batches', worker, n_workers,
                    ntmp, len(self.coarse), len(self.batches))

    @classmethod
    def from_cli(cls, args, live_bank, worker, n_workers):
        """Build the worker's filter from the pycbc_live options."""
        from pycbc.waveform.bank import RatioFilterBank
        seglen = args.ratio_segment_length
        delta_f = 1.0 / seglen
        flen_coarse = int(seglen * args.sample_rate) // 2 + 1
        flen_fine = round(args.multiband_frequency / delta_f)
        bank = RatioFilterBank(args.bank_file,
                               filter_length=(flen_fine, flen_coarse),
                               delta_f=delta_f, dtype=numpy.complex64)
        bank.delta_f = delta_f
        return cls(bank, live_bank.table, worker, n_workers,
                   args.snr_threshold, args.chisq_bins,
                   args.multiband_frequency,
                   fir_length=args.fir_length,
                   batch_size=args.template_batch_size,
                   gate_safety_sigma=args.gate_safety_sigma,
                   gate_extra_margin=args.gate_extra_margin,
                   band_ratio_threshold=args.band_ratio_threshold,
                   snr_abort_threshold=args.snr_abort_threshold,
                   newsnr_threshold=args.newsnr_threshold,
                   max_triggers_in_batch=args.max_triggers_in_batch)

    def process_data(self, data_reader):
        """Filter the newest stride of one detector's data with every
        template of this worker. Returns the trigger dict LiveBatchMatchedFilter
        returns, or False if the stride is aborted."""
        ow = data_reader.overwhitened_data(self.delta_f)
        psd = ow.psd
        # The live buffer hands out overwhitened data (strain / PSD); the
        # ratio filter divides by the PSD itself, so undo it once here
        stilde = ow * psd
        stilde.psd = psd

        nfull = (len(stilde) - 1) * 2
        valid_end = int(nfull - data_reader.trim_padding)
        valid_start = int(valid_end
                          - data_reader.blocksize * data_reader.sample_rate)
        valid = slice(valid_start, valid_end)

        # The loudest peak of each fine template in the stride
        best = {}
        for b in self.batches:
            eng = self.engine
            eng.hnorms_low = b.hnorms_low if b.hnorms_psd is psd else None
            ids, t_idxs, snrs, snrs_low, snrs_high, _, h_norm = \
                eng.process_segment(
                    stilde, psd, self.coarse[b.coarse_index], b.filters_f,
                    b.counts, numpy.arange(len(b.fine_ids)),
                    valid_slice=valid,
                    lowband_templates=(b.templates, b.templates2),
                    scale=(b.scale_low, b.scale_high))
            b.hnorms_low, b.hnorms_psd = eng.hnorms_low, psd
            for j in range(len(ids)):
                fid = int(b.fine_ids[ids[j]])
                s = abs(snrs[j])
                if fid not in best or s > abs(best[fid][1]):
                    best[fid] = (int(t_idxs[j]), snrs[j], snrs_low[j],
                                 snrs_high[j], float(h_norm[j]), b, ids[j])

        if self.snr_abort_threshold is not None and best and \
                max(abs(v[1]) for v in best.values()) > \
                self.snr_abort_threshold:
            logger.info("We are seeing some *really* high SNRs, let's "
                        "assume they aren't signals and just give up")
            return False

        fids = numpy.array(list(best.keys()), dtype=numpy.int64)
        if len(fids) and self.max_triggers_in_batch:
            order = numpy.argsort([abs(best[f][1]) for f in fids])[::-1]
            fids = fids[order[:self.max_triggers_in_batch]]

        n = len(fids)
        snr = numpy.array([best[f][1] for f in fids], dtype=numpy.complex64)
        tidx = numpy.array([best[f][0] for f in fids], dtype=numpy.int64)
        result = {key: self.param_table[key][fids]
                  for key in self.param_table.dtype.names}
        result['snr'] = abs(snr)
        result['coa_phase'] = numpy.angle(snr)
        result['end_time'] = data_reader.start_time + \
            (tidx - valid_start) / float(data_reader.sample_rate)
        result['template_id'] = fids.astype(numpy.uint64)
        result['sigmasq'] = numpy.array([best[f][4] for f in fids],
                                        dtype=numpy.float32)
        result['chisq'] = numpy.zeros(n, dtype=numpy.float32)
        result['chisq_dof'] = numpy.zeros(n, dtype=numpy.uint32)
        # No sine-Gaussian veto for the ratio filter yet: values below the
        # sgveto threshold leave newsnr_sgveto unchanged
        result['sg_chisq'] = numpy.ones(n, dtype=numpy.float32)

        if self.power_chisq.do:
            for i, f in enumerate(fids):
                c, d = self._chisq(best[f], stilde, psd)
                result['chisq'][i] = c / d
                result['chisq_dof'][i] = d

        if self.newsnr_threshold:
            keep = ranking.newsnr(result['snr'], result['chisq']) >= \
                self.newsnr_threshold
            result = {k: v[keep] for k, v in result.items()}
        return result

    def _chisq(self, trig, stilde, psd):
        """Power chisq over the band below the split, plus a 2 dof term for
        the consistency of the low and high band SNRs (as offline)."""
        t_idx, snr, snr_low, snr_high, h_norm, b, local = trig
        if abs(snr_high) / abs(snr_low) >= self.band_ratio_threshold:
            # Most of the SNR is above the split: assume a glitch
            return 1e9, 1
        template = b.templates[local]
        kmax = len(template)
        corr = template.conj() * stilde[:kmax] / psd[:kmax] \
            * (4 * stilde.delta_f / h_norm ** 0.5)
        tids = numpy.array([t_idx], ndmin=1) * kmax / ((len(stilde) - 1) * 2)
        chi, dof = self.power_chisq.values(
            corr, numpy.array([snr_low], ndmin=1), 1, psd[:kmax], tids,
            template)
        chi = float(chi[0]) + abs(snr_high) ** 2 + abs(snr_low) ** 2 \
            - abs(snr) ** 2
        return chi, int(dof[0]) + 2
