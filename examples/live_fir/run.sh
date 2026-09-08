#!/bin/bash
# Ratio-Filter Dechirping in PyCBC Live: run the same short analysis with the
# standard matched filter and with --ratio-bank-file, then check that the
# recovered SNR / triggers agree within the mismatch bound.
#
# arXiv:2601.18835 (Nitz, Kacanja & Soni).

set -e
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE

gps_start_time=1272790000
gps_end_time=1272790512
f_min=20

LDIR=$(dirname -- "${BASH_SOURCE[0]}")

echo ">> [`date`] Building template banks"
python $LDIR/make_grid_bank.py --fine-out bank_fine.hdf --coarse-out bank_coarse.hdf \
    --approximant IMRPhenomD --f-lower $f_min

echo ">> [`date`] Building FIR ratio bank"
python $LDIR/make_fir_bank.py \
    --coarse-bank bank_coarse.hdf --fine-bank bank_fine.hdf --output-file bank_fir.hdf \
    --sample-rate 2048 --f-low $f_min --delta-f 1.0 --n-taps 201 \
    --decimation 2 --ridge 1e-3 --search-depth 4 -n 2 \
    --psd-model aLIGOZeroDetHighPower --approximant IMRPhenomD

echo ">> [`date`] Checking the reconstruction math"
python $LDIR/test_ratio_math.py bank_fir.hdf --n 10 --approximant IMRPhenomD \
    --psd-model aLIGOZeroDetHighPower

# simulated strain for two detectors
if [[ ! -d ./strain ]]; then
    echo ">> [`date`] Generating simulated strain"
    for det_seed in "H1 1234" "L1 2345"; do
        set -- $det_seed
        mkdir -p strain/$1
        pycbc_condition_strain \
            --fake-strain aLIGOMidLowSensitivityP1200087 --fake-strain-seed $2 \
            --output-strain-file "strain/$1/$1-SIMULATED_STRAIN-{start}-{duration}.gwf" \
            --gps-start-time $gps_start_time --gps-end-time $gps_end_time \
            --sample-rate 2048 --low-frequency-cutoff 10 \
            --channel-name $1:SIMULATED_STRAIN --frame-duration 32
    done
fi

live_opts=(
    --sample-rate 2048
    --low-frequency-cutoff $f_min
    --max-length 256
    --approximant IMRPhenomD
    --chisq-bins "0.72*get_freq('fIMRPhenomDPeak',params.mass1,params.mass2,params.spin1z,params.spin2z)**0.7"
    --snr-threshold 4.0
    --newsnr-threshold 4.0
    --max-triggers-in-batch 30
    --store-loudest-index 20
    --analysis-chunk 8
    --highpass-frequency 13 --highpass-bandwidth 5 --highpass-reduction 200
    --psd-samples 20 --psd-segment-length 4 --psd-inverse-length 3.5
    --max-psd-abort-distance 600 --min-psd-abort-distance 20
    --psd-abort-difference .15 --psd-recalculate-difference .01
    --trim-padding .5 --store-psd
    --increment-update-cache H1:strain/H1 L1:strain/L1
    --frame-src H1:"strain/H1/*" L1:"strain/L1/*"
    --frame-read-timeout 10
    --channel-name H1:SIMULATED_STRAIN L1:SIMULATED_STRAIN
    --processing-scheme cpu:1
    --fftw-measure-level 0
    --increment 8
    --max-batch-size 16777216
    --sngl-ranking newsnr_sgveto
    --sgchisq-snr-threshold 4
    --sgchisq-locations "mtotal>40:20-30,20-45,20-60,20-75,20-90,20-105,20-120"
    --ranking-statistic quadsum
    --enable-background-estimation
    --background-ifar-limit 100
    --timeslide-interval 0.1
    --pvalue-combination-livetime 0.0005
    --ifar-double-followup-threshold 0.0001
    --ifar-upload-threshold 0.0001
    --src-class-mchirp-to-delta 0.01
    --src-class-eff-to-lum-distance 0.74899
    --src-class-lum-distance-to-delta -0.51557 -0.32195
    --round-start-time 4
    --start-time $gps_start_time
    --end-time $gps_end_time
)

echo ">> [`date`] Standard PyCBC Live"
rm -rf output_standard && mkdir -p output_standard
mpirun --oversubscribe -n 3 python -m mpi4py `which pycbc_live` \
    --bank-file bank_fine.hdf --output-path output_standard "${live_opts[@]}"

echo ">> [`date`] Ratio-Filter PyCBC Live"
rm -rf output_fir && mkdir -p output_fir
mpirun --oversubscribe -n 3 python -m mpi4py `which pycbc_live` \
    --ratio-bank-file bank_fir.hdf --fir-length 32768 \
    --template-normalization-method parseval \
    --output-path output_fir "${live_opts[@]}"

echo ">> [`date`] Comparing"
python $LDIR/compare_triggers.py output_standard output_fir --fir-bank bank_fir.hdf
