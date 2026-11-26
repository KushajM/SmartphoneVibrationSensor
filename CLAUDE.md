# Smartphone-Based Vibration Signature Analysis — Project Context

UG Electrical Engineering course project (Project Computation Course, Jadavpur University).
Goal: classify 4 vibration conditions of a household wall fan using only a smartphone's
built-in MEMS accelerometer. The student is comfortable with ML/deep learning, so explanations
can be brief, but the deliverables are graded on signal-processing correctness, so keep the DSP rigorous
and well-commented.

## Current status

- Data collected: 80 trials (20/class), one session on 2026-11-04, unpacked to `data/raw/`
  (Phyphox zips in `data/zips/`). Sensor clock 400.108 Hz, no dropped samples.
- Full pipeline written **in Python** (student's instruction); the final model is also ported to
  MATLAB R2018a (see below). `py run_all.py` regenerates everything in `results/`
  (Phase 1 audit, Phase 2 features/plots, noise calibration, Phase 3 classification + robustness);
  `results/SUMMARY.md` is the generated overview; `py -m src.predict <zip>` is the demo.
- Final model: two-stage logistic regression (n_harm detector + f0), noise-augmented training.
- MATLAB port in `matlab/` (vib_predict, vib_demo) verified on R2018a against Python: identical on all 80 trials.
  Claude can run MATLAB via bash: `/d/Apps/MATLAB/bin/win64/MATLAB.exe -nosplash -nodesktop -minimize -wait -logfile <log> -r "...; exit"`.
- Final report: `report/Final_Report.docx` (student-edited, the reference). LaTeX versions on Overleaf: IEEE ieeecolor two-column (git https://git.overleaf.com/6af34053d8956f6d6ceb4d15, current; `report/to_latex_ieee.py` + `report/fig_paper.py` for the landscape-page robustness figures) and an older spr-sola version (git https://git.overleaf.com/6aed6dbcb561ae4d9287d551; `report/to_latex.py`); both regenerated from `report/dump_docx.py` output. Code is public at https://github.com/KushajM/SmartphoneVibrationSensor (zips moved to data/zips/).
- Still open: blade count / rpm check, second recording session.

## Language

The brief says **preferably MATLAB**, with Python (NumPy, SciPy, Matplotlib) allowed.
Ask whether MATLAB is installed locally before choosing. If it is, write MATLAB; otherwise Python.
Don't mix the two in the final deliverable.

## Experimental setup (decided)

- **Source:** wall-mounted fan at home, oscillation (sweep) turned OFF for all trials.
- **Classes (4):** `off`, `speed1`, `speed2`, `speed3` (regulator positions).
- **Phone mounting:** taped flat to the wall right next to the fan's mounting bracket; orientation
  fixed and outline marked. Phone axes therefore are wall-relative, not table-relative.
- **App:** Phyphox → Raw Sensors → **Acceleration (with g)**. Gravity is deliberately left in;
  removing it is part of the graded preprocessing.
- **Trials:** 15 per class = **60 total** (brief requires 50+ total). Each is a separate recording
  (timed run, 2 s start delay, 7 s duration) — trim to the central 5 s during preprocessing.
- **Recording order:** round-robin (off → s1 → s2 → s3, ×15) so slow drift isn't confounded with class.
- **Expected frequencies:** fan rotation roughly 15–25 Hz; blade-pass = rotation × number of blades.
  Actual rpm unknown — the student may estimate it (e.g. slow-motion video). Phone sampling is roughly
  100–400 Hz, so watch the Nyquist limit and mention aliasing if any harmonics approach fs/2.

## Data format

Each Phyphox CSV export is a zip containing `Raw Data.csv` (plus a `meta/` folder).
Expected columns (verify headers against an actual file):

```
Time (s), Acceleration x (m/s^2), Acceleration y (m/s^2), Acceleration z (m/s^2), Absolute acceleration (m/s^2)
```

Phone timestamps are **not uniformly spaced** — always check `diff(t)` and resample onto a
uniform grid before any FFT/CWT.

Zips are named `<class>_<NN>.zip` (e.g. `speed2_07.zip`) and get unpacked to:

```
data/
  zips/                     # raw Phyphox exports
  raw/<class>/<class>_NN.csv
log/recording_log.csv       # trial, class, time, notes (written by hand during recording)
```

Suggested code/output layout:

```
src/        load_trials, preprocess, features, plots, classify
results/    feature_table.csv, figures/, confusion_matrix.png
report/
```

## Assignment brief (weekly plan + deliverables)

**Week 1 — Acquisition (3 h).** Verify sampling-rate stability, mount rigidly with consistent
orientation, record 50+ trials of 4 classes, 5 s each, tri-axial.
*Deliverable:* raw CSV dataset + short note on mounting method and achieved sampling rate.

**Weeks 2–3 — Signal processing & features (3–6 h).**
- Preprocess: remove gravity (high-pass or mean subtraction), resample to a uniform time base.
- Time-domain features: RMS, peak-to-peak, crest factor, kurtosis.
- Frequency domain: FFT and CWT to find the dominant frequency; compare with expected rotational
  speed. Optional: envelope spectrum (bearing-type faults).
*Deliverable:* feature table (4 classes) + FFT and wavelet plots.

**Weeks 3–4 — Classification & report (3–6 h).**
- Simple threshold-based classifier or basic ML (kNN or logistic regression) on **2–3 features**.
- Validate on held-out trials; report accuracy and confusion matrix.
*Deliverable:* final report + code, 5-minute in-class demo.

## Rubric

| Component | Weight |
|---|---|
| Data quality & documentation | 20% |
| Signal processing correctness | 30% |
| Feature discriminability / analysis depth | 25% |
| Classification result & validation | 15% |
| Report / presentation clarity | 10% |

Signal-processing correctness is the largest component — justify filter choices (type, order,
cutoff), windowing, resampling method, and FFT resolution in comments and in the report.

## Pipeline expectations

1. **Load & audit:** per file, compute fs = 1/median(dt), dt jitter, duration; flag files that are
   short, gappy, or have outlier fs. Produce a summary table for the Week 1 note.
2. **Preprocess:** resample to a common uniform fs → trim to central 5 s → remove gravity
   (prefer a zero-phase high-pass, e.g. low-order Butterworth with `filtfilt`, cutoff ~1–2 Hz;
   state the choice). Keep per-axis signals and also the vector magnitude.
3. **Features (one row per trial):** RMS, peak-to-peak, crest factor, kurtosis (per axis and/or
   magnitude), dominant FFT frequency, and its amplitude. Add class label and trial ID columns.
4. **Plots:** one representative trial per class: time series, FFT (with expected rotation and
   blade-pass frequencies marked), CWT scalogram. Feature scatter plots to show separability.
5. **Classify:** pick 2–3 features with the best separation. Use a stratified **trial-level** split
   (e.g. 10 train / 5 test per class) or cross-validation. Report accuracy and confusion matrix.

## Pitfalls to avoid

- **Leakage:** never split windows of the same trial across train and test. Each trial is one sample.
- **Don't FFT non-uniform data** — resample first.
- The `off` class is ambient/background vibration; it may be near-silent, so the noise floor matters.
- Fan spin-up/spin-down transients should already be excluded by the 15 s settle time and the trim,
  but check the first and last second of a few trials.
- Keep the code reproducible: one entry-point script that runs the whole pipeline from `data/raw`
  to `results/`.
