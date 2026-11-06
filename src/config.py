"""Central configuration. Every tunable constant lives here with its justification, so the
report can quote one place for all signal-processing choices."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
ZIP_DIRS = [ROOT / "data" / "zips", ROOT]      # Phyphox exports (only meta/ is read from them)
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
TAB = RESULTS / "tables"
MODEL_DIR = RESULTS / "model"
CAL_FILE = RESULTS / "noise_calibration.json"

CLASSES = ["off", "speed1", "speed2", "speed3"]
CLASS_COLORS = {"off": "0.25", "speed1": "tab:blue", "speed2": "tab:orange", "speed3": "tab:red"}
SEED = 20261104

# ------------------------------------------------------------------ Phase 1: preprocessing
FS = 400.0          # common analysis rate [Hz]. The sensor's hardware clock runs at 400.108 Hz;
                    # every trial is mapped onto exactly 400.000 Hz so all spectra share one axis.
TRIM_S = 5.0        # analysed window, centred in each 7 s recording (1 s discarded at each end)
SINC_HALF = 16      # half-width (input samples) of the Kaiser-windowed-sinc interpolator
SINC_BETA = 8.0     # Kaiser beta -> stop-band ~ -80 dB; passband flat to ~0.45 fs.
                    # (Linear interpolation was REJECTED: at f/fs = 0.27 its gain at a half-sample
                    # offset is cos(pi f/fs) = 0.65, and because 400.108 Hz and 400 Hz grids slide
                    # past each other this produced a fake, start-locked 35 % amplitude decay.)
HP_ORDER = 4        # gravity removal: Butterworth high-pass run forward-backward (sosfiltfilt):
HP_FC = 2.0         # zero phase, effective 8th-order magnitude, -6 dB at 2 Hz. 2 Hz is far above
                    # gravity (DC) and tape creep (< 0.01 Hz) and > 10x below the lowest fan line
                    # (~32 Hz), so the fan band is untouched (< 0.01 dB loss above 10 Hz).
HAMPEL_HALF = 5     # Hampel despiker: +-5 samples (27.5 ms window) ...
HAMPEL_NSIG = 6.0   # ... replaces samples > 6 robust sigma from the local median. On the clean
                    # data it flags ~0 samples (Gaussian false-alarm rate ~2e-9).

# ------------------------------------------------------------------ Phase 2: spectra / features
WELCH_NPERSEG = 1024   # mean-averaged Welch (plots, textbook features): 2.56 s segments,
                       # Hann, 50 % overlap, df = 0.39 Hz  (class lines are >= 3.6 Hz apart)
ROB_NPERSEG = 512      # median-averaged Welch (robust features): 1.28 s segments, 75 % overlap
                       # -> 12 segments in 5 s; median rejects transients present in < half of
                       # them. df = 0.78 Hz, refined to sub-bin accuracy by parabolic interpolation
FLOOR_HALFWIDTH_HZ = 10.0   # local noise floor = running median over +-10 Hz
F0_RANGE = (28.0, 42.0)     # physical prior on the fan fundamental (observed 31.7-37.2 Hz, +-10 %
                            # headroom). Excludes the k=2 / k=4 aliases of the dominant 3*f0 line,
                            # which otherwise cause harmonic-number errors under heavy noise.
F0_STEP = 0.02
K_MAX = 4                   # harmonics in the comb (4*f0 <= 168 Hz < Nyquist for every class)
SNR_CAP_DB = 30.0           # one harmonic contributes at most 30 dB to the comb score
STRONG_DB = 6.0             # harmonic counted as "present" above 6 dB
MAINS_HZ = 50.0             # Indian grid
MAINS_MASK_HW = 1.0         # +-1 Hz around 50/100/150 Hz never counts as a fan harmonic
BAND = (5.0, 190.0)         # search band for "dominant frequency" features
CWT_CYCLES = 24.0           # Morlet: sigma_f = f / 24 (4.2 Hz at 100 Hz), sigma_t = 38 ms at 100 Hz
