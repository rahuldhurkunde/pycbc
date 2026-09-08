##################################################
Ratio-Filter Dechirping matched filter (PyCBC Live)
##################################################

.. contents::

This example exercises the hierarchical "Ratio-Filter Dechirping" matched
filter in ``pycbc_live`` (``--ratio-bank-file``), the low-latency form of the
method of Nitz, Kacanja & Soni (arXiv:2601.18835).

Instead of matched-filtering every template in a dense bank, ``pycbc_live``
filters only a sparse *coarse* reference bank and reconstructs each dense
*fine* template's SNR time series by convolving a short (real, ~200-tap) FIR
filter with the coarse reference's SNR::

    (s|h)(t) = F^-1[ h~(f) / h~_ref(f) ]  *  (s|h)_ref(t)

The ratio ``h~/h~_ref`` is slowly varying, so its inverse transform is a
compact time-domain kernel.  This removes the "re-filter the whole template
every increment" redundancy of the standard low-latency matched filter.

Running
=======

::

    bash run.sh

which

#. ``make_grid_bank.py`` -- writes a small deterministic aligned-spin BBH
   ``bank_fine.hdf`` and a sparser ``bank_coarse.hdf``.
#. ``make_fir_bank.py`` -- for every fine template, picks the best nearby
   coarse reference and least-squares fits a short FIR filter to their
   frequency-domain ratio, producing ``bank_fir.hdf`` (fine bank at the root,
   coarse bank + per-reference FIR taps under ``fir_data/``).
#. ``test_ratio_math.py`` -- for a sample of fine templates, checks that the
   ratio reconstruction of an injected signal's SNR matches a direct matched
   filter (peak SNR ratio ~ 1, timing exact).
#. a plain ``pycbc_live`` run over simulated two-detector strain
   (``output_standard``);
#. the same run with ``--ratio-bank-file bank_fir.hdf`` (``output_fir``);
#. ``compare_triggers.py`` -- matches triggers by detector + template + time
   and reports the SNR-difference spread against the mismatch bound
   ``sqrt(2*(1-min_match))``.

Interpreting the comparison
==========================

The FIR reconstruction runs in the same precision as the standard filter, so
the SNR-difference spread should sit near the mismatch-induced fluctuation
bound ``sqrt(2*(1-min_match))`` (a few times it, at most -- ``--margin``),
trigger times should agree to ~1 sample, and the reduced-chisq values (which
the FIR engine computes by regenerating each surviving fine template) should
track the standard ones.

Notes
=====

- This is a *correctness* example.  The ratio stage here is plain NumPy FFTs;
  the paper's throughput gain needs cache-resident kernels, not yet wired in.
- ``make_fir_bank.py`` is a fast, simplified stand-in for a production
  ratio-bank builder (nearest-neighbour coarse selection, single ridge fit,
  no tap-count escalation).
