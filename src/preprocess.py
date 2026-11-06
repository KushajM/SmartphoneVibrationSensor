"""Phase 1 preprocessing:  timestamps -> hardware-clock slots -> Hampel despike ->
band-limited resampling to exactly 400 Hz -> zero-phase high-pass (gravity removal) ->
central 5 s window.

Order matters:
  * despike BEFORE resampling: a sinc interpolator would smear a single corrupt sample into
    32 ringing samples;
  * high-pass the FULL 7 s record, THEN trim: filtfilt start-up transients (~0.5 s for a
    2 Hz corner) fall in the discarded 1 s margins.
"""
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from . import config as C


@dataclass
class Prep:
    x: np.ndarray            # N x 3 gravity-free acceleration, FS, central window [m/s^2]
    valid: np.ndarray        # N bool: False where the sample lies in/near a dropped-sample gap
    fs_in: float             # true device rate (from the hardware-clock fit)
    gravity: np.ndarray      # mean raw acceleration vector (gravity estimate) [m/s^2]
    n_despiked: int          # samples replaced by the Hampel filter
    x_full: np.ndarray       # gravity-free, full record (for plots / edge checks)
    i0: int                  # start index of the window inside x_full


# ------------------------------------------------------------------ timing
def regularise(t, a):
    """Assign samples to integer slots of the sensor clock and fit its rate.

    Returns xs (n_slots x 3; dropped slots bridged linearly and flagged), fs_in, missing.
    The phone's timestamps scatter by ~0.1 us around a perfect 400.108 Hz grid, so treating the
    samples as uniform at the fitted rate is exact for all practical purposes; it is also the
    only correct model for a FIFO-buffered MEMS sensor whose ODR is set by its own oscillator.
    """
    dt_med = np.median(np.diff(t))
    slots = np.round((t - t[0]) / dt_med).astype(np.int64)
    keep = np.concatenate([[True], np.diff(slots) > 0])          # drop duplicate slots
    slots, tt, aa = slots[keep], t[keep], a[keep]
    slope, _ = np.polyfit(slots, tt, 1)
    n = int(slots[-1]) + 1
    missing = np.ones(n, bool)
    missing[slots] = False
    idx = np.arange(n)
    xs = np.column_stack([np.interp(idx, slots, aa[:, k]) for k in range(aa.shape[1])])
    return xs, 1.0 / slope, missing


# ------------------------------------------------------------------ band-limited interpolation
def sinc_eval(xs, pos, ratio=1.0, half=C.SINC_HALF, beta=C.SINC_BETA):
    """Evaluate a uniformly sampled signal at fractional sample positions `pos` with a
    Kaiser-windowed sinc kernel. ratio < 1 lowers the kernel cut-off to ratio*Nyquist
    (anti-alias filtering when the output rate is lower than the input rate)."""
    n_in = xs.shape[0]
    hw = int(np.ceil(half / ratio))
    i0 = np.floor(pos).astype(np.int64)
    k = np.arange(-hw + 1, hw + 1)
    idx = i0[:, None] + k[None, :]
    d = pos[:, None] - idx
    w = ratio * np.sinc(ratio * d) * np.i0(beta * np.sqrt(np.clip(1 - (d / hw) ** 2, 0, None))) / np.i0(beta)
    w /= w.sum(axis=1, keepdims=True)                               # exact unity DC gain
    idx = np.clip(idx, 0, n_in - 1)                                 # edge samples repeated
    mu = xs.mean(axis=0)
    return np.einsum("ij,ijk->ik", w, xs[idx] - mu) + mu


def resample_uniform(xs, fs_in, fs_out=C.FS):
    dur = (xs.shape[0] - 1) / fs_in
    t_out = np.arange(0.0, dur, 1.0 / fs_out)
    return sinc_eval(xs, t_out * fs_in, ratio=min(1.0, fs_out / fs_in)), t_out


def resample_linear(xs, fs_in, fs_out=C.FS):
    """Kept ONLY to demonstrate the interpolation artefact in the report."""
    t_in = np.arange(xs.shape[0]) / fs_in
    t_out = np.arange(0.0, t_in[-1], 1.0 / fs_out)
    return np.column_stack([np.interp(t_out, t_in, xs[:, k]) for k in range(xs.shape[1])]), t_out


# ------------------------------------------------------------------ despiking
def hampel(x, half=C.HAMPEL_HALF, nsig=C.HAMPEL_NSIG):
    """Hampel identifier: flag |x - local median| > nsig * 1.4826 * local MAD and replace the
    sample by the local median. A record-wide robust sigma is used as a floor so that coarsely
    quantised signals (local MAD = 0) are not 'despiked' into staircases."""
    xp = np.pad(x, ((half, half), (0, 0)), mode="reflect")
    win = sliding_window_view(xp, 2 * half + 1, axis=0)             # n x 3 x w
    med = np.median(win, axis=-1)
    mad = np.median(np.abs(win - med[..., None]), axis=-1)
    resid = x - med
    floor = 1.4826 * np.median(np.abs(resid - np.median(resid, axis=0)), axis=0)
    sigma = np.maximum(1.4826 * mad, floor[None, :])
    flag = np.abs(resid) > nsig * np.maximum(sigma, 1e-12)
    out = np.where(flag, med, x)
    return out, flag


# ------------------------------------------------------------------ gravity removal
def highpass(x, fs=C.FS):
    sos = signal.butter(C.HP_ORDER, C.HP_FC, "highpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=0)


def hp_response(f, fs=C.FS):
    """Magnitude of the forward-backward filter (= |H|^2 of the single pass)."""
    sos = signal.butter(C.HP_ORDER, C.HP_FC, "highpass", fs=fs, output="sos")
    _, h = signal.sosfreqz(sos, worN=f, fs=fs)
    return np.abs(h) ** 2


# ------------------------------------------------------------------ full chain
def preprocess(t, a, *, method="sinc", despike=True, trim=C.TRIM_S, fs=C.FS):
    xs, fs_in, missing = regularise(t, a)
    n_spk = 0
    if despike:
        xs, flag = hampel(xs)
        n_spk = int(flag.sum())
    if method == "sinc":
        y, t_out = resample_uniform(xs, fs_in, fs)
    elif method == "linear":
        y, t_out = resample_linear(xs, fs_in, fs)
    else:
        raise ValueError(method)
    # a target sample is invalid if any dropped slot lies within +-2 input samples
    bad = np.convolve(missing.astype(float), np.ones(5), mode="same") > 0
    valid = ~bad[np.clip(np.round(t_out * fs_in).astype(int), 0, len(bad) - 1)]
    g = y.mean(axis=0)
    xf = highpass(y, fs)
    n = int(round(trim * fs))
    i0 = max((len(xf) - n) // 2, 0)
    return Prep(x=xf[i0:i0 + n], valid=valid[i0:i0 + n], fs_in=fs_in, gravity=g,
                n_despiked=n_spk, x_full=xf, i0=i0)
