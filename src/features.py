"""Phase 2 features: one row per trial.

Groups
  time domain   RMS, peak-to-peak, crest factor, kurtosis on x, y, z and on the principal
                vibration axis pc1 (orientation-invariant), plus the vector RMS
  FFT           dominant frequency + amplitude (z axis = textbook; vector spectrum = invariant)
  Welch         dominant frequency, spectral centroid, spectral flatness, band-energy ratio
  comb          harmonic-comb SNR, fundamental f0, harmonicity (# strong harmonics), line SNR
  CWT           time-averaged Morlet scalogram peak (feature table only)
"""
import warnings

import numpy as np
from scipy import signal, stats
from scipy.ndimage import median_filter

from . import config as C


def _pow2_floor(n):
    return 1 << int(np.floor(np.log2(max(int(n), 2))))


def _parabolic(f, P, k):
    """Sub-bin peak location from a parabola through log-power at bins k-1, k, k+1."""
    if k <= 0 or k >= len(P) - 1:
        return float(f[k])
    y0, y1, y2 = np.log(np.maximum(P[k - 1:k + 2], 1e-30))
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
    return float(f[k] + np.clip(d, -0.5, 0.5) * (f[1] - f[0]))


def _peak(f, P, band):
    m = np.where((f >= band[0]) & (f <= band[1]))[0]
    k = m[np.argmax(P[m])]
    return _parabolic(f, P, k), float(P[k])


# ------------------------------------------------------------------ spectra
def amp_spectrum(x, fs=C.FS):
    """Single-sided amplitude spectrum, Hann window, coherent-gain corrected: a sinusoid of
    amplitude A reads A at its peak. Zero-padded x2-4 (interpolates the spectrum only; the
    resolution stays ~1.5 fs/N = 0.3 Hz for N = 2000)."""
    n = x.shape[0]
    w = signal.windows.hann(n, sym=False)
    nfft = 1 << (int(np.ceil(np.log2(n))) + 1)
    xx = x - x.mean(axis=0)
    X = np.fft.rfft(xx * (w[:, None] if x.ndim == 2 else w), n=nfft, axis=0)
    return np.fft.rfftfreq(nfft, 1 / fs), 2 * np.abs(X) / w.sum()


def welch_axes(x, fs=C.FS, nperseg=C.WELCH_NPERSEG):
    """Mean-averaged Welch PSD per axis [(m/s^2)^2/Hz], Hann, 50 % overlap."""
    nper = min(nperseg, _pow2_floor(len(x)))
    return signal.welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2, axis=0)


def robust_psd(x, valid=None, fs=C.FS, nperseg=C.ROB_NPERSEG, median=True, axes=None):
    """Gap-aware, median-averaged Welch PSD summed over axes.

    * Summing per-axis PSDs gives the PSD of the acceleration VECTOR, which does not depend on
      how the phone is rotated (unlike any single axis).
    * Median across segments ignores events that occupy fewer than half the segments
      (footsteps, knocks, spikes); segments overlapping dropped samples are skipped.
    * The median is biased low for noise bins by a constant factor; it cancels in the SNR
      (both line and floor come from the same estimate), so no correction is applied.
    Returns f, P, local floor (running median +-10 Hz), SNR in dB.
    """
    xx = x if axes is None else x[:, axes]
    if xx.ndim == 1:
        xx = xx[:, None]
    n = len(xx)
    nper = min(nperseg, _pow2_floor(n // 2))
    hop = max(nper // 4, 1)
    starts = np.arange(0, n - nper + 1, hop)
    if valid is not None and not valid.all():
        ok = np.array([valid[s:s + nper].mean() >= 0.95 for s in starts])
        if ok.sum() >= 3:
            starts = starts[ok]
    w = signal.windows.hann(nper, sym=False)
    segs = np.stack([xx[s:s + nper] for s in starts])
    segs = segs - segs.mean(axis=1, keepdims=True)
    X = np.fft.rfft(segs * w[None, :, None], axis=1)
    Pseg = (np.abs(X) ** 2).sum(axis=2) * (2.0 / (fs * (w ** 2).sum()))
    P = np.median(Pseg, axis=0) if (median and len(starts) >= 3) else Pseg.mean(axis=0)
    f = np.fft.rfftfreq(nper, 1 / fs)
    size = 2 * int(round(C.FLOOR_HALFWIDTH_HZ / (f[1] - f[0]))) + 1
    floor = median_filter(P, size=size, mode="nearest")
    snr = 10 * np.log10(np.maximum(P, 1e-30) / np.maximum(floor, 1e-30))
    return f, P, floor, snr


# ------------------------------------------------------------------ harmonic comb
def comb_features(f, P, snr, fs_in, mains_mask=True, f0_range=C.F0_RANGE):
    """Fit a harmonic comb k*f0 (k = 1..4) to the SNR spectrum.

    score(f0) = mean over usable harmonics of clip(SNR(k f0), 0, 30 dB)
      - usable: below 0.45 x the DEVICE rate (no trust near/above its Nyquist) and not inside
        a +-1 Hz mains band (50/100/150 Hz hum is fixed-frequency, never a fan harmonic); a
        candidate may lose at most one harmonic to the mask
      - the cap stops one strong foreign line from dominating the score
    comb_snr = best score (detection statistic, dB); f0 = refined fundamental (SNR-weighted
    average of parabolic harmonic peaks / k); n_harm = harmonics >= 6 dB; line_snr = strongest.
    """
    s = snr.copy()
    if mains_mask:
        for m in (1, 2, 3):
            s[np.abs(f - m * C.MAINS_HZ) <= C.MAINS_MASK_HW] = np.nan
    fmax = min(0.45 * fs_in, 0.95 * f[-1])
    grid = np.arange(f0_range[0], f0_range[1], C.F0_STEP)
    ks = np.arange(1, C.K_MAX + 1)
    H = grid[:, None] * ks[None, :]
    inb = H < fmax
    S = np.where(inb, np.clip(np.interp(H, f, s), 0, C.SNR_CAP_DB), np.nan)
    n_in = inb.sum(axis=1)
    n_mask = (inb & np.isnan(S)).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        score = np.nanmean(S, axis=1)
    ok = (n_in - n_mask >= 1) & (n_mask <= 1) & np.isfinite(score)
    if not ok.any():
        return dict(comb_snr=0.0, f0=float(np.mean(f0_range)), n_harm=0, line_snr=0.0)
    score = np.where(ok, score, -np.inf)
    i = int(np.argmax(score))
    f0c = grid[i]
    est, wts = [], []
    for k in ks:
        if k * f0c >= fmax:
            continue
        j = np.where((np.abs(f - k * f0c) <= 1.2) & np.isfinite(s))[0]
        if j.size == 0:
            continue
        jj = j[np.argmax(P[j])]
        est.append(_parabolic(f, P, jj) / k)
        wts.append(max(s[jj], 0.0) + 1e-6)
    f0 = float(np.average(est, weights=wts)) if est else float(f0c)
    Si = S[i][np.isfinite(S[i])]
    return dict(comb_snr=float(score[i]), f0=f0, n_harm=int((Si >= C.STRONG_DB).sum()),
                line_snr=float(Si.max()) if Si.size else 0.0)


# ------------------------------------------------------------------ CWT
def cwt_morlet(sig, fs, freqs, cycles=C.CWT_CYCLES):
    """Continuous wavelet transform with a complex Morlet wavelet, computed in the frequency
    domain: at centre frequency fc the wavelet is a Gaussian band-pass of sigma_f = fc/cycles
    (constant-Q). Amplitude-normalised: a sinusoid of amplitude A gives |W| = A at its own
    frequency. Zero-padding to >= 2N avoids circular wrap-around."""
    n = len(sig)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.fft(sig - np.mean(sig), nfft)
    nu = np.fft.fftfreq(nfft, 1 / fs)
    W = np.empty((len(freqs), n), complex)
    for i, fc in enumerate(freqs):
        G = np.where(nu > 0, 2.0 * np.exp(-0.5 * ((nu - fc) / (fc / cycles)) ** 2), 0.0)
        W[i] = np.fft.ifft(X * G)[:n]
    return W


def cwt_fdom(sig, fs=C.FS, band=(20.0, 190.0)):
    freqs = np.arange(band[0], band[1], 0.5)
    W = cwt_morlet(sig, fs, freqs)
    edge = int(2 * C.CWT_CYCLES / (2 * np.pi * band[0]) * fs)      # 2 sigma_t cone of influence
    edge = min(edge, len(sig) // 4)
    pw = np.median(np.abs(W[:, edge:len(sig) - edge]) ** 2, axis=1)
    return _parabolic(freqs, pw, int(np.argmax(pw)))


# ------------------------------------------------------------------ time domain
def pca_axis(x):
    """Projection on the direction of largest vibration variance (rotation-invariant scalar)."""
    w, v = np.linalg.eigh(np.cov(x.T))
    return x @ v[:, -1]


def time_feats(s, tag):
    rms = float(np.sqrt(np.mean(s ** 2)))
    flat = np.std(s) < 1e-9          # e.g. an axis frozen by a very coarse ADC step
    return {f"rms_{tag}": rms, f"p2p_{tag}": float(s.max() - s.min()),
            f"crest_{tag}": 0.0 if flat else float(np.abs(s).max() / rms),
            f"kurt_{tag}": 3.0 if flat else float(stats.kurtosis(s, fisher=False))}


# ------------------------------------------------------------------ full feature vector
FEATURE_NAMES = (
    [f"{s}_{a}" for a in ("x", "y", "z", "pc1") for s in ("rms", "p2p", "crest", "kurt")]
    + ["rms_vec", "fft_fdom_z", "fft_amp_z", "fft_fdom_vec", "fft_amp_vec",
       "welch_fdom", "spec_centroid", "spec_flatness", "band_ratio",
       "comb_snr", "f0", "n_harm", "line_snr"])

DEFAULT_OPTS = dict(median=True, mains_mask=True, f0_prior=True, axis_sum=True)


def extract(prep, cwt=False, opts=None):
    o = dict(DEFAULT_OPTS, **(opts or {}))
    x, v = prep.x, prep.valid
    xv = x[v] if v.mean() > 0.5 else x
    F = {}
    pc = pca_axis(xv)
    for tag, s in (("x", xv[:, 0]), ("y", xv[:, 1]), ("z", xv[:, 2]), ("pc1", pc)):
        F.update(time_feats(s, tag))
    F["rms_vec"] = float(np.sqrt(np.mean((xv ** 2).sum(axis=1))))
    f, A = amp_spectrum(x)
    F["fft_fdom_z"], F["fft_amp_z"] = _peak(f, A[:, 2], C.BAND)
    F["fft_fdom_vec"], F["fft_amp_vec"] = _peak(f, np.sqrt((A ** 2).sum(axis=1)), C.BAND)
    fw, Pw = welch_axes(x)
    Ps = Pw.sum(axis=1)
    F["welch_fdom"], _ = _peak(fw, Ps, C.BAND)
    m = (fw >= 20) & (fw <= 190)
    F["spec_centroid"] = float((fw[m] * Ps[m]).sum() / Ps[m].sum())
    F["spec_flatness"] = float(np.exp(np.mean(np.log(np.maximum(Ps[m], 1e-30)))) / Ps[m].mean())
    F["band_ratio"] = float(Ps[(fw >= 90) & (fw <= 115)].sum() / Ps[(fw >= C.BAND[0]) & (fw <= C.BAND[1])].sum())
    fr, Pr, _, snr = robust_psd(x, v, median=o["median"], axes=None if o["axis_sum"] else [2])
    F.update(comb_features(fr, Pr, snr, prep.fs_in, mains_mask=o["mains_mask"],
                           f0_range=C.F0_RANGE if o["f0_prior"] else (20.0, 60.0)))
    if cwt:
        F["cwt_fdom"] = cwt_fdom(pc)
    return F


def vector(F, names=FEATURE_NAMES):
    v = np.array([F[n] for n in names], dtype=float)
    return np.nan_to_num(v, nan=0.0, posinf=1e6, neginf=-1e6)     # last line of defence
