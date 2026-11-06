# Smartphone-Based Vibration Signature Analysis

This project classifies the operating state of a household wall fan (**off / speed 1 / speed 2 / speed 3**) from a smartphone's MEMS accelerometer. The data are Phyphox recordings, *Acceleration with g*, from a TDK ICM-45621 sampling at 400 Hz.

UG Electrical Engineering course project, Jadavpur University. The full report (LaTeX, on Overleaf) documents every design choice. This repository holds the complete code, the data and every generated result.

## Results at a glance

| | |
|---|---|
| Final model | two-stage logistic regression: *is the fan on?* (`n_harm`), then *which speed?* (`f0`), trained with calibrated noise augmentation |
| Leave-one-round-out | 80/80 correct (95 % CI 95.5–100 %) |
| Repeated 5-fold CV (×10) | 100.0 ± 0.0 % |
| Calibrated realistic household noise | 100 % for every family |
| MATLAB port vs Python | identical predictions on 80/80 recordings (max Δf0 = 1.7·10⁻⁸ Hz) |

## Layout

```
data/zips/              80 raw Phyphox exports (<class>_NN.zip)
data/raw/<class>/       the same recordings unpacked to CSV
src/                    Python pipeline
  config.py             every constant, with its justification
  load_trials.py        loading + acquisition audit
  preprocess.py         clock fit, Hampel despiking, Kaiser-sinc resampling, zero-phase high-pass
  features.py           time / FFT / Welch / CWT features, harmonic comb
  calibrate_noise.py    noise-model calibration against the recordings
  noise_models.py       22 calibrated noise generators + augmentation
  ml.py                 models, nested feature selection, CV, statistics
  phase1_audit.py, phase2_features.py, phase3_classify.py,
  phase3_robustness.py, phase3_selection_study.py   phase drivers + figures
  predict.py            command-line demo classifier
  report.py             writes results/SUMMARY.md
matlab/                 MATLAB R2018a port of the final model (vib_predict, vib_demo, ...)
results/                feature table, tables/*.csv, figures/*.png, model/final_model.json, logs
run_all.py              single entry point: data/raw -> results/
```

## Running it

```bash
pip install -r requirements.txt
python run_all.py                     # full pipeline (~8 min on 16 cores)
python run_all.py --quick             # robustness study with 2 instead of 5 repeats
python -m src.predict data/zips/*.zip --csv out.csv            # demo classifier
python -m src.predict data/zips/speed2_05.zip --noise footsteps 0.3   # with calibrated noise added
```

MATLAB (R2018a or newer, Signal Processing Toolbox):

```matlab
cd matlab
[label, probs] = vib_predict('../data/zips/speed2_05.zip')
vib_demo        % classify all 80 exports and verify against the Python reference
```

All randomness is seeded, so every run reproduces bit for bit.
