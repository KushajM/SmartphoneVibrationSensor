"""Corruption ("noise") models for a wall-mounted smartphone accelerometer.

Every model takes the RAW recording (timestamps, 3-axis acceleration incl. gravity) and returns
what the phone would have reported, so the complete pipeline (timing fit, despiking,
resampling, high-pass, features) is exercised. Signature: fn(rng, t, a, level, cal) -> (t, a).

`cal` comes from calibrate_noise.py (measured on the 80 real recordings + datasheets +
literature). It provides the measured noise floor, the measured background spectral shape
(used as the sensor-response FIR that every physically-generated noise passes through), the
fan-vibration power and line amplitude used as references, and the realistic level bands.

Families are grouped; when a family is under test, its whole GROUP is removed from the
training augmentation (leave-one-group-out), so robustness is always measured on noise
the classifier never saw during training.
"""
import numpy as np
from scipy import signal

from .preprocess import regularise, sinc_eval

G0 = 9.80665


# ------------------------------------------------------------------ helpers
def _shape(x, cal):
    """Pass white noise through the measured sensor/background response (linear-phase FIR)."""
    h = np.asarray(cal["sensor_fir"])
    return signal.oaconvolve(x, h[:, None], mode="same", axes=0)


def shaped_noise(rng, shape, asd, cal):
    """Gaussian noise with one-sided ASD `asd` [m/s^2/sqrt(Hz)] in the passband, spectrally
    shaped like the real fan-off recordings."""
    w = rng.normal(0.0, asd * np.sqrt(cal["fs_ref"] / 2.0), size=shape)
    return _shape(w, cal)


def _unit(rng, zbias=0.0):
    v = rng.normal(size=3)
    v[2] += zbias * (1 if v[2] >= 0 else -1)
    return v / np.linalg.norm(v)


def _modal(rng, tt, f_ranges, zeta, rel):
    """Impulse response of a lightly damped structure: sum of decaying sinusoids, unit peak."""
    y = np.zeros_like(tt)
    for (lo, hi), r in zip(f_ranges, rel):
        fm, z = rng.uniform(lo, hi), rng.uniform(*zeta)
        wn = 2 * np.pi * fm
        y += r * rng.uniform(0.7, 1.3) * np.exp(-z * wn * tt) * np.sin(wn * np.sqrt(1 - z * z) * tt + rng.uniform(0, 2 * np.pi))
    return y / (np.abs(y).max() + 1e-12)


def _events(rng, t, times, peak, kernel, zbias, cal):
    out = np.zeros((len(t), 3))
    for t0 in times:
        i0 = int(np.searchsorted(t, t0))
        tt = t[i0:] - t0
        tt = tt[tt < 1.5]
        if tt.size < 8:
            continue
        out[i0:i0 + tt.size] += (kernel(tt) * peak * rng.uniform(0.6, 1.0))[:, None] * _unit(rng, zbias)[None, :]
    return _shape(out, cal)


# ------------------------------------------------------------------ electronic / sensor noise
def gaussian(rng, t, a, snr_db, cal):
    """Textbook additive WHITE Gaussian noise (AWGN), i.i.d. per axis, at a given SNR relative
    to the fan's broadband vibration power (3 axes, measured)."""
    if not np.isfinite(snr_db):
        return t, a
    sigma = np.sqrt(cal["fan_power_3axis"] / 10 ** (snr_db / 10) / 3)
    return t, a + rng.normal(0.0, sigma, a.shape)


def johnson_nyquist(rng, t, a, temp_c, cal):
    """Thermal (Johnson-Nyquist / thermo-mechanical Brownian) noise of the MEMS proof mass.
    Fluctuation-dissipation theorem (Gabrielson 1993): S_a = 4 k_B T omega0 / (m Q) - white,
    Gaussian, proportional to absolute temperature. Worst case assumed: the whole measured
    floor at 25 C is thermal, so heating to T adds S_floor * (T/298.15 K - 1)."""
    extra = cal["floor_psd"] * max((temp_c + 273.15) / 298.15 - 1.0, 0.0)
    return t, a + shaped_noise(rng, a.shape, np.sqrt(extra), cal) if extra > 0 else a


def jn_proofmass(rng, t, a, r, cal):
    """Same thermal law, but a sensor whose proof mass (x quality factor) is r times smaller:
    S_a scales by r. Maps directly onto cheaper/smaller accelerometers (r = (ND/ND_this)^2)."""
    extra = cal["floor_psd"] * max(r - 1.0, 0.0)
    return t, (a + shaped_noise(rng, a.shape, np.sqrt(extra), cal)) if extra > 0 else a


def pink(rng, t, a, fc, cal):
    """Flicker (1/f, pink) noise whose PSD equals the measured white floor at corner fc."""
    if fc <= 0:
        return t, a
    n = len(a)
    w = rng.normal(0.0, np.sqrt(cal["floor_psd"] * cal["fs_ref"] / 2.0), size=(n, 3))
    W = np.fft.rfft(w, axis=0)
    f = np.fft.rfftfreq(n, 1 / cal["fs_ref"])
    g = np.sqrt(fc / np.maximum(f, 0.05))
    g[0] = 0.0
    return t, a + _shape(np.fft.irfft(W * g[:, None], n=n, axis=0), cal)


def brownian(rng, t, a, sigma_rw, cal):
    """Brownian (red, 1/f^2) noise = bias random walk, sigma_rw in m/s^2 per sqrt(s)."""
    if sigma_rw <= 0:
        return t, a
    dt = 1.0 / cal["fs_ref"]
    return t, a + np.cumsum(rng.normal(0.0, sigma_rw * np.sqrt(dt), a.shape), axis=0)


def popcorn(rng, t, a, amp, cal, rate=5.0):
    """Burst / popcorn noise (random telegraph signal): the offset jumps between two levels
    `amp` apart at Poisson times (5 switches/s per axis)."""
    if amp <= 0:
        return t, a
    out = np.zeros_like(a)
    T = t[-1] - t[0]
    for k in range(3):
        sw = np.sort(rng.uniform(t[0], t[-1], rng.poisson(rate * T)))
        state = rng.choice([-1.0, 1.0]) * (-1.0) ** np.searchsorted(sw, t)
        out[:, k] = 0.5 * amp * state
    return t, a + _shape(out, cal)


def salt_pepper(rng, t, a, density, cal):
    """Impulse (salt-and-pepper) noise: a fraction `density` of samples per axis is replaced by
    +full-scale (salt) or -full-scale (pepper) - corrupt words on the sensor bus / saturation
    glitches. Full scale from the phone's metadata (+-8 g = 78.45 m/s^2)."""
    if density <= 0:
        return t, a
    out = a.copy()
    m = rng.random(a.shape) < density
    s = rng.random(a.shape) < 0.5
    out[m & s] = cal["sensor_range"]
    out[m & ~s] = -cal["sensor_range"]
    return t, out


def speckle(rng, t, a, var, cal):
    """Speckle (multiplicative) noise, MATLAB imnoise definition J = I + n.*I with n uniform,
    zero mean, variance `var`, applied to the vibration (gravity excluded)."""
    if var <= 0:
        return t, a
    mu = a.mean(axis=0)
    d = a - mu
    n = (rng.random(a.shape) - 0.5) * np.sqrt(12.0 * var)
    return t, mu + d * (1.0 + n)


def quantize(rng, t, a, lsb, cal):
    """Coarser ADC: values rounded to multiples of `lsb` [m/s^2] (gravity included, as in a
    real sensor)."""
    if lsb <= 0:
        return t, a
    return t, np.round(a / lsb) * lsb


# ------------------------------------------------------------------ environmental disturbances
def footsteps(rng, t, a, peak, cal):
    """Person walking nearby: heel strikes at 1.6-2.2 steps/s (+-20 ms scatter), each exciting
    floor/wall modes 8-90 Hz with 3-8 % damping. `peak` = peak structural acceleration."""
    if peak <= 0:
        return t, a
    fstep = rng.uniform(1.6, 2.2)
    times = np.arange(t[0] + rng.uniform(0, 1 / fstep), t[-1], 1 / fstep)
    times = times + rng.normal(0, 0.02, size=times.size)
    k = lambda tt: _modal(rng, tt, [(8, 20), (20, 45), (45, 90)], (0.03, 0.08), [1.0, 0.6, 0.3])
    return t, a + _events(rng, t, times, peak, k, 1.0, cal)


def knocks(rng, t, a, peak, cal):
    """Door slam / knock / dropped object: 1-3 impacts, 3 ms half-sine contact + 4-6 lightly
    damped (1-4 %) modes between 10 and 180 Hz."""
    if peak <= 0:
        return t, a
    times = rng.uniform(t[0] + 0.5, t[-1] - 0.5, size=rng.integers(1, 4))

    def k(tt):
        nm = int(rng.integers(4, 7))
        y = _modal(rng, tt, [(10, 180)] * nm, (0.01, 0.04), [1.0] * nm)
        y = y + np.where(tt < 0.003, np.sin(np.pi * tt / 0.003), 0.0)
        return y / np.abs(y).max()
    return t, a + _events(rng, t, times, peak, k, 0.5, cal)


def _tones(rng, t, a, freqs, amps, ratio, cal):
    d = _unit(rng, 0.5)
    s = sum(A * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) for f, A in zip(freqs, amps))
    s = s / np.sqrt(np.mean(s ** 2))
    return t, a + (ratio * cal["fan_line_rms"] * s)[:, None] * d[None, :]


def mains_hum(rng, t, a, ratio, cal):
    """Magnetostriction hum (transformers, ballasts, motor stator) at 2 x 50 = 100 Hz plus 50 and
    150 Hz. Fixed frequency. `ratio` = hum RMS / fan dominant-line RMS."""
    return (t, a) if ratio <= 0 else _tones(rng, t, a, [100.0, 50.0, 150.0], [1.0, 0.3, 0.3], ratio, cal)


def appliance(rng, t, a, ratio, cal):
    """Another machine, e.g. a 2-pole fridge/AC compressor (2850-2970 rpm -> 47.5-49.5 Hz) and
    harmonics. Its 2nd harmonic (95-99 Hz) lands on speed1's dominant line."""
    if ratio <= 0:
        return t, a
    fc = rng.uniform(47.5, 49.5)
    return _tones(rng, t, a, [fc, 2 * fc, 3 * fc], [0.6, 1.0, 0.4], ratio, cal)


# ------------------------------------------------------------------ mounting / operating point
def orientation(rng, t, a, deg, cal):
    """Phone re-taped at a different angle: rotate the whole 3-axis signal (gravity included)
    by `deg` about a random axis (Rodrigues formula)."""
    if deg <= 0:
        return t, a
    th, k = np.deg2rad(deg), _unit(rng)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
    return t, a @ R.T


def coupling(rng, t, a, g, cal):
    """Looser tape / different wall spot: structural vibration reaches the sensor with gain g.
    The sensor's own noise does not scale, so it is topped back up (shaped, measured level)."""
    if g >= 1:
        return t, a
    mu = a.mean(axis=0)
    top = shaped_noise(rng, a.shape, cal["floor_asd"] * np.sqrt(max(1 - g * g, 0.0)), cal)
    return t, mu + g * (a - mu) + top


def speed_shift(rng, t, a, pct, cal):
    """Supply-voltage change: same regulator position, different slip -> every fan line moves
    by (1 + d), d = +-pct % (random sign). Band-limited time-scaling of the record."""
    if pct <= 0:
        return t, a
    d = rng.choice([-1.0, 1.0]) * pct / 100.0
    xs, fs_in, _ = regularise(t, a)
    n = len(xs)
    y = sinc_eval(xs, np.clip(np.arange(n) * (1 + d), 0, n - 1))
    return t[0] + np.arange(n) / fs_in, y


# ------------------------------------------------------------------ acquisition
def dropouts(rng, t, a, pct, cal):
    """OS scheduling hiccups: gaps of 20-150 ms totalling `pct` % of the samples."""
    if pct <= 0:
        return t, a
    keep = np.ones(len(t), bool)
    target = int(pct / 100 * len(t))
    fs = cal["fs_ref"]
    while (~keep).sum() < target:
        L = int(rng.uniform(0.02, 0.15) * fs)
        i = int(rng.integers(1, len(t) - L - 1))
        keep[i:i + L] = False
    return t[keep], a[keep]


def _decimate(t, a, fs_new, ratio_fn):
    xs, fs_in, _ = regularise(t, a)
    if fs_new >= fs_in - 1:
        return t, a
    n_out = int((len(xs) - 1) * fs_new / fs_in) + 1
    y = sinc_eval(xs, np.arange(n_out) * fs_in / fs_new, ratio=ratio_fn(fs_new / fs_in))
    return t[0] + np.arange(n_out) / fs_new, y


def lower_fs(rng, t, a, fs_new, cal):
    """Phone/app sampling at a lower output data rate, with the sensor's own digital anti-alias
    low-pass (cut-off 0.45 fs_new) - the normal case for modern MEMS."""
    return _decimate(t, a, fs_new, lambda r: 0.9 * r)


def lower_fs_aliased(rng, t, a, fs_new, cal):
    """Same rate, but NO anti-alias filter: lines above fs_new/2 fold back."""
    return _decimate(t, a, fs_new, lambda r: 1.0)


# ------------------------------------------------------------------ calibrated scenario mixes
def _u(rng, hi):
    return rng.uniform(0.0, hi) if hi > 0 else 0.0


def _household(rng, t, a, sev, cal):
    """This phone, this mount, same day: everything that the calibration says happens."""
    B = cal["bands"]
    t, a = speed_shift(rng, t, a, abs(rng.normal(0, cal["speed_sameday_sigma_pct"] * sev)), cal)
    t, a = coupling(rng, t, a, 1 - _u(rng, (1 - B["coupling"][0]) * sev), cal)
    t, a = speckle(rng, t, a, _u(rng, B["speckle"][1] * sev), cal)
    if rng.random() < 0.5:
        t, a = footsteps(rng, t, a, _u(rng, B["footsteps"][1] * sev), cal)
    if rng.random() < 0.3:
        t, a = knocks(rng, t, a, _u(rng, B["knocks"][1] * sev), cal)
    t, a = mains_hum(rng, t, a, _u(rng, B["mains_hum"][1] * sev), cal)
    t, a = johnson_nyquist(rng, t, a, rng.uniform(15, 45), cal)
    t, a = pink(rng, t, a, _u(rng, B["pink"][1] * sev), cal)
    t, a = brownian(rng, t, a, _u(rng, B["brownian"][1] * sev), cal)
    return t, a


def mix_same_day(rng, t, a, sev, cal):
    return (t, a) if sev <= 0 else _household(rng, t, a, sev, cal)


def mix_other_day(rng, t, a, sev, cal):
    """+ phone re-taped (orientation) and a different day's supply voltage (speed shift)."""
    if sev <= 0:
        return t, a
    t, a = orientation(rng, t, a, min(_u(rng, cal["bands"]["orientation"][1] * sev), 180), cal)
    t, a = speed_shift(rng, t, a, _u(rng, cal["speed_day_pct"] * sev), cal)
    return _household(rng, t, a, sev, cal)


def mix_other_phone(rng, t, a, sev, cal):
    """+ a different phone: a configuration drawn from the datasheet catalogue (accelerometer
    noise density, ADC step, 400 Hz or the Android-12 200 Hz app cap) + re-taping."""
    if sev <= 0:
        return t, a
    cat = cal["phone_catalogue"]
    t, a = orientation(rng, t, a, min(_u(rng, cal["bands"]["orientation"][1] * sev), 180), cal)
    t, a = _household(rng, t, a, sev, cal)
    t, a = jn_proofmass(rng, t, a, float(rng.choice(cat["psd_mult"])), cal)
    t, a = lower_fs(rng, t, a, float(rng.choice(cat["fs"])), cal)
    t, a = quantize(rng, t, a, float(rng.choice(cat["lsb"])), cal)
    return t, a


# ------------------------------------------------------------------ registry
# levels: swept at test time (first entry = clean). aug: training-augmentation draw.
SPEC = {
    "gaussian":        dict(fn=gaussian, levels=[np.inf, 20, 10, 0, -5, -10, -15, -20, -25],
                            unit="SNR re fan vibration [dB]", group="gaussian-type",
                            aug=("lin", -12.0, 20.0)),
    "johnson_nyquist": dict(fn=johnson_nyquist, levels=[25, 45, 85, 125, 250, 500],
                            unit="sensor temperature [deg C]", group="gaussian-type", aug=("lin", 15.0, 85.0)),
    "jn_proofmass":    dict(fn=jn_proofmass, levels=[1, 3, 10, 30, 100, 300],
                            unit="thermal-noise PSD multiplier (1/mQ)", group="gaussian-type",
                            aug=("log", 1.0, 60.0)),
    "pink":            dict(fn=pink, levels=[0, 1, 3, 10, 30, 100],
                            unit="1/f corner frequency [Hz]", group="gaussian-type", aug=("log", 0.1, 30.0)),
    "brownian":        dict(fn=brownian, levels=[0, 1e-3, 1e-2, 3e-2, 0.1, 0.3],
                            unit="random walk [m/s^2/sqrt(s)]", group="drift", aug=("log", 1e-4, 3e-2)),
    "popcorn":         dict(fn=popcorn, levels=[0, 0.003, 0.01, 0.03, 0.1, 0.3],
                            unit="burst step [m/s^2]", group="drift", aug=("log", 1e-3, 0.05)),
    "salt_pepper":     dict(fn=salt_pepper, levels=[0, 1e-5, 1e-4, 1e-3, 1e-2, 3e-2],
                            unit="fraction of samples", group="impulse", aug=("log", 1e-5, 1e-2)),
    "speckle":         dict(fn=speckle, levels=[0, 0.01, 0.04, 0.1, 0.25, 0.5, 1.0],
                            unit="multiplicative variance", group="multiplicative", aug=("lin", 0.0, 0.5)),
    "coupling":        dict(fn=coupling, levels=[1, 0.8, 0.6, 0.4, 0.25, 0.15],
                            unit="mount gain", group="multiplicative", aug=("lin", 0.3, 1.0)),
    "quantize":        dict(fn=quantize, levels=[0, 0.0024, 0.0048, 0.0096, 0.019, 0.038, 0.077, 0.153],
                            unit="ADC step [m/s^2]", group="adc", aug=("log", 0.0006, 0.04)),
    "footsteps":       dict(fn=footsteps, levels=[0, 0.01, 0.03, 0.1, 0.3, 1.0],
                            unit="peak accel [m/s^2]", group="mechanical", aug=("log", 0.003, 0.2)),
    "knocks":          dict(fn=knocks, levels=[0, 0.03, 0.1, 0.3, 1.0, 3.0],
                            unit="peak accel [m/s^2]", group="mechanical", aug=("log", 0.01, 0.5)),
    "mains_hum":       dict(fn=mains_hum, levels=[0, 0.3, 1, 2, 4, 8],
                            unit="hum RMS / fan-line RMS", group="tonal", aug=("lin", 0.0, 4.0)),
    "appliance":       dict(fn=appliance, levels=[0, 0.1, 0.3, 1, 2, 4],
                            unit="compressor RMS / fan-line RMS", group="tonal", aug=("lin", 0.0, 1.0)),
    "orientation":     dict(fn=orientation, levels=[0, 5, 15, 45, 90, 180],
                            unit="rotation [deg]", group="geometry", aug=("lin", 0.0, 180.0)),
    "speed_shift":     dict(fn=speed_shift, levels=[0, 0.5, 1, 1.5, 2, 3, 4, 6],
                            unit="fan speed change [+-%]", group="speed", aug=("lin", 0.0, 1.5)),
    "dropouts":        dict(fn=dropouts, levels=[0, 2, 5, 10, 20, 30],
                            unit="samples lost [%]", group="timing", aug=("lin", 0.0, 10.0)),
    "lower_fs":        dict(fn=lower_fs, levels=[400, 300, 250, 200, 150, 100],
                            unit="device rate [Hz], anti-aliased", group="rate",
                            aug=("choice", [400.0, 300.0, 250.0, 200.0])),
    "lower_fs_aliased": dict(fn=lower_fs_aliased, levels=[400, 300, 250, 200, 150, 100],
                             unit="device rate [Hz], NO anti-alias", group="rate", aug=None),
    "mix_same_day":    dict(fn=mix_same_day, levels=[0, 0.5, 1, 2, 4],
                            unit="severity (1 = calibrated realistic)", group="mix", aug=None),
    "mix_other_day":   dict(fn=mix_other_day, levels=[0, 0.5, 1, 2, 4],
                            unit="severity (1 = calibrated realistic)", group="mix", aug=None),
    "mix_other_phone": dict(fn=mix_other_phone, levels=[0, 0.5, 1, 2, 4],
                            unit="severity (1 = calibrated realistic)", group="mix", aug=None),
}
# physically sensible order when several corruptions are combined in one augmented copy
ORDER = ["speed_shift", "orientation", "coupling", "speckle", "footsteps", "knocks", "mains_hum",
         "appliance", "gaussian", "johnson_nyquist", "jn_proofmass", "pink", "brownian", "popcorn",
         "lower_fs", "quantize", "salt_pepper", "dropouts"]
AUG_POOL = [n for n in ORDER if SPEC[n]["aug"] is not None]


def _draw(rng, spec):
    kind = spec[0]
    if kind == "lin":
        return float(rng.uniform(spec[1], spec[2]))
    if kind == "log":
        return float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
    return float(rng.choice(spec[1]))


def apply(name, rng, t, a, level, cal):
    return SPEC[name]["fn"](rng, t, a, level, cal)


GROUPS = sorted({v["group"] for v in SPEC.values()} - {"mix"})      # bit order for group masks


def augment(rng, t, a, cal, exclude_groups=(), return_mask=False):
    """One training copy: 1-3 randomly chosen corruption families (not in excluded groups),
    each at a random level from its augmentation range, applied in physical order.
    return_mask: also return a bitmask of the noise GROUPS used (for worst-group selection)."""
    pool = [n for n in AUG_POOL if SPEC[n]["group"] not in exclude_groups]
    m = int(rng.integers(1, 4))
    chosen = set(rng.choice(pool, size=min(m, len(pool)), replace=False))
    mask = 0
    for n in ORDER:
        if n in chosen:
            t, a = SPEC[n]["fn"](rng, t, a, _draw(rng, SPEC[n]["aug"]), cal)
            mask |= 1 << GROUPS.index(SPEC[n]["group"])
    return (t, a, mask) if return_mask else (t, a)
