"""Calibrate every noise model against real measurements.

Evidence hierarchy (stated per family in results/tables/noise_calibration.csv):
  measured     - from the 80 recordings themselves (fan-off trials = the real noise floor,
                 fan-on trials = real line amplitude/frequency variability, timing, gravity)
  datasheet    - accelerometer datasheets / ADC arithmetic for other phones
  literature   - standards and published measurements (supply voltage, floor vibration, Android)
  assumption   - plausible value with no measurement available (flagged)
  uncalibrated - amplitude unknowable without new recordings; reported as a stress test only
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, signal, stats
from scipy.ndimage import median_filter

from . import config as C
from . import features as F
from . import load_trials as L
from . import noise_models as N
from . import preprocess as P

K_B = 1.380649e-23
G0 = 9.80665

# accelerometer noise density [ug/sqrt(Hz)] of parts used in phones
DATASHEET_ND = [
    ("TDK ICM-45605 (same family as this phone's ICM-45621)", 70.0),
    ("TDK ICM-42688-P / ST LSM6DSO", 70.0),
    ("TDK ICM-40609-D / ICM-42670-P", 100.0),
    ("InvenSense MPU-6000/6050 (older phones)", 400.0),
]
# ADC step = full-scale span / 2^bits
ADC_EXAMPLES = [("16-bit, +-2 g", 4 * G0 / 2 ** 16), ("16-bit, +-4 g", 8 * G0 / 2 ** 16),
                ("16-bit, +-8 g", 16 * G0 / 2 ** 16), ("16-bit, +-16 g", 32 * G0 / 2 ** 16),
                ("12-bit, +-8 g", 16 * G0 / 2 ** 12), ("10-bit, +-4 g", 8 * G0 / 2 ** 10)]


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else (1e300 if o > 0 else -1e300)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def load_cal():
    cal = json.loads(C.CAL_FILE.read_text())
    for k, v in cal["bands"].items():                     # restore +-inf
        if v is not None:
            cal["bands"][k] = [np.inf if x >= 1e299 else (-np.inf if x <= -1e299 else x) for x in v]
    return cal


def calibrate(trials=None, figs=True, verbose=True):
    trials = trials or L.load_all()
    C.FIG.mkdir(parents=True, exist_ok=True)
    C.TAB.mkdir(parents=True, exist_ok=True)
    off = [t for t in trials if t.cls == "off"]
    on = [t for t in trials if t.cls != "off"]
    prep = {t.name: P.preprocess(t.t, t.a) for t in trials}
    feat = {t.name: F.extract(prep[t.name]) for t in trials}
    fs_ref = float(np.mean([p.fs_in for p in prep.values()]))
    ev = {}

    # ---------------------------------------------------------------- 1. sensor noise floor
    S_off = []
    for t in off:
        f, S = F.welch_axes(prep[t.name].x)
        S_off.append(S)
    S_off = np.mean(S_off, axis=0)                                   # mean PSD per axis
    pl = (f >= 10) & (f <= 60)
    floor_psd_axis = np.median(S_off[pl], axis=0)
    floor_psd = float(floor_psd_axis.mean())
    floor_asd = float(np.sqrt(floor_psd))
    nd_ug = floor_asd / G0 * 1e6
    floor_sigma = float(np.median([np.std(prep[t.name].x, axis=0) for t in off]))
    shape = median_filter(S_off.mean(axis=1) / floor_psd, size=5, mode="nearest")
    shape[f < 10] = 1.0                                               # below: HP region, use flat
    shape = np.clip(shape, 1e-5, 4.0)
    fir = signal.firwin2(129, f, np.sqrt(shape), fs=C.FS)
    b_eff = float(np.trapz(shape, f))
    z = np.concatenate([(prep[t.name].x / prep[t.name].x.std(axis=0)).ravel() for t in off])
    kurt_trials = [stats.kurtosis(prep[t.name].x[:, k], fisher=False) for t in off for k in range(3)]
    cc = np.mean([np.abs(np.corrcoef(prep[t.name].x.T)[np.triu_indices(3, 1)]).mean() for t in off])
    ev["floor"] = dict(nd_ug=nd_ug, sigma=floor_sigma, kurt=float(stats.kurtosis(z, fisher=False)),
                       skew=float(stats.skew(z)), kurt_range=(float(np.min(kurt_trials)), float(np.max(kurt_trials))),
                       mean_abs_axis_corr=float(cc), b_eff=b_eff)

    cal = dict(fs_ref=fs_ref, sensor_fir=fir, floor_psd=floor_psd, floor_asd=floor_asd,
               floor_sigma=floor_sigma, sensor_range=float(off[0].meta.get("range", 78.45)))

    # validation of the synthetic floor model: generate, run through the SAME pipeline, compare
    rng = np.random.default_rng(C.SEED)
    S_syn, sig_syn, kurt_syn = [], [], []
    for _ in range(20):
        n = 2800
        ts = np.arange(n) / fs_ref
        a_s = np.array([0.0, 9.785, 0.0]) + N.shaped_noise(rng, (n, 3), floor_asd, cal)
        ps = P.preprocess(ts, a_s)
        S_syn.append(F.welch_axes(ps.x)[1])
        sig_syn.append(np.std(ps.x, axis=0))
        kurt_syn.append(stats.kurtosis(ps.x[:, 0], fisher=False))
    S_syn = np.mean(S_syn, axis=0)
    band = (f >= 5) & (f <= 190)
    ev["floor"]["synthetic_sigma_ratio"] = float(np.median(sig_syn) / floor_sigma)
    ev["floor"]["synthetic_psd_rms_db_err"] = float(np.sqrt(np.mean((10 * np.log10(S_syn.mean(1)[band] / S_off.mean(1)[band])) ** 2)))

    # ---------------------------------------------------------------- 2. 1/f (pink) corner
    S_lf = []
    for t in off:
        xs, fsi, _ = P.regularise(t.t, t.a)
        xs = signal.detrend(xs, axis=0)
        f_lf, S = signal.welch(xs, fs=fsi, nperseg=2048, noverlap=1024, axis=0)
        S_lf.append(S.mean(axis=1))
    S_lf = np.mean(S_lf, axis=0)
    m = (f_lf >= 0.2) & (f_lf <= 20)
    fun = lambda ff, lw, lp: np.log(np.exp(lw) + np.exp(lp) / ff)
    (lw, lp), _ = optimize.curve_fit(fun, f_lf[m], np.log(S_lf[m]), p0=[np.log(floor_psd), np.log(floor_psd)])
    fc_pink = float(np.exp(lp) / np.exp(lw))
    ev["pink"] = dict(fc=fc_pink, S_white=float(np.exp(lw)))

    # ---------------------------------------------------------------- 3. bias random walk
    off_sorted = sorted(off, key=lambda t: t.meta.get("start_epoch", 0))
    gm = np.array([t.a.mean(axis=0) for t in off_sorted])
    te = np.array([t.meta.get("start_epoch", 0) for t in off_sorted])
    dmu, dte = np.diff(gm, axis=0), np.diff(te)[:, None]
    rw = float(np.sqrt(np.mean(dmu ** 2 / dte)))                      # m/s^2 per sqrt(s)
    ev["brownian"] = dict(sigma_rw=rw, drift_total=float(np.abs(gm[-1] - gm[0]).max()),
                          span_min=float((te[-1] - te[0]) / 60))

    # ---------------------------------------------------------------- 4. popcorn bound
    sig_d, max_ratio = [], []
    for t in off:
        xs, fsi, _ = P.regularise(t.t, t.a)
        xs = signal.detrend(xs, axis=0)
        nb = int(0.05 * fsi)
        blk = xs[: (len(xs) // nb) * nb].reshape(-1, nb, 3).mean(axis=1)
        d = np.diff(blk, axis=0)
        s = 1.4826 * np.median(np.abs(d), axis=0)
        sig_d.append(s.mean())
        max_ratio.append((np.abs(d) / s).max())
    rtn_bound = float(5 * np.median(sig_d))
    ev["popcorn"] = dict(step_sigma=float(np.median(sig_d)), max_step_over_sigma=float(np.max(max_ratio)),
                         bound=rtn_bound)

    # ---------------------------------------------------------------- 5. impulses (salt & pepper)
    n_tot = int(sum(len(t.t) * 3 for t in trials))
    n_spk = int(sum(prep[t.name].n_despiked for t in trials))
    p_bound = (n_spk if n_spk else 3.0) / n_tot                       # rule of three if none seen
    ev["salt_pepper"] = dict(samples=n_tot, spikes=n_spk, rate_bound=p_bound)

    # ---------------------------------------------------------------- 6. fan line (on trials)
    rows = []
    for t in on:
        p = prep[t.name]
        f0 = feat[t.name]["f0"]
        f3 = 3 * f0
        pc = F.pca_axis(p.x)
        sos = signal.butter(4, [f3 - 3, f3 + 3], "bandpass", fs=C.FS, output="sos")
        y = signal.sosfiltfilt(sos, pc)
        e = np.abs(signal.hilbert(y))[100:-100]
        pn = float(np.interp(f3, f, S_off.mean(axis=1)) * 6.0)        # in-band noise power
        pline = max(np.var(y) - pn, 1e-12)
        cv_raw = float(e.std() / e.mean())
        cv_noise = float(np.sqrt(pn / (2 * pline)))
        rows.append(dict(name=t.name, cls=t.cls, f0=f0, line_rms=float(np.sqrt(pline)), cv_raw=cv_raw,
                         cv_mult=float(np.sqrt(max(cv_raw ** 2 - cv_noise ** 2, 0.0))),
                         epoch=t.meta.get("start_epoch", 0.0),
                         power3=float((p.x ** 2).sum(axis=1).mean())))
    off_power3 = float(np.median([(prep[t.name].x ** 2).sum(axis=1).mean() for t in off]))
    fan_power = float(np.median([r["power3"] for r in rows]) - off_power3)
    fan_line_rms = float(np.median([r["line_rms"] for r in rows]))
    cv_mult = float(np.median([r["cv_mult"] for r in rows]))
    cv_raw_med = float(np.median([r["cv_raw"] for r in rows]))
    # white speckle of variance v puts v*3/(fs/2) of its variance inside the +-3 Hz band around
    # the line, i.e. envelope CV^2. The measured wobble is (almost) fully explained by additive
    # noise (cv_mult ~ 0), so the RAW CV is used as a conservative upper bound.
    speckle_v = max(cv_mult, cv_raw_med) ** 2 * (fs_ref / 2) / 3.0
    line_cv_cls = {c: float(np.std([r["line_rms"] for r in rows if r["cls"] == c]) /
                            np.mean([r["line_rms"] for r in rows if r["cls"] == c])) for c in C.CLASSES[1:]}
    g_lo = 1 - 2 * max(line_cv_cls.values())
    f0s = {c: np.array([r["f0"] for r in rows if r["cls"] == c]) for c in C.CLASSES[1:]}
    f0_sig_pct = {c: float(v.std() / v.mean() * 100) for c, v in f0s.items()}
    drift = {}
    for c in C.CLASSES[1:]:
        rr = [r for r in rows if r["cls"] == c]
        hrs = (np.array([r["epoch"] for r in rr]) - rr[0]["epoch"]) / 3600
        fv = np.array([r["f0"] for r in rr])
        drift[c] = float(np.polyfit(hrs, fv / fv.mean() * 100, 1)[0])  # % per hour
    sameday_sigma = float(np.median(list(f0_sig_pct.values())))
    ev["fan"] = dict(fan_power_3axis=fan_power, native_snr_db=float(10 * np.log10(fan_power / off_power3)),
                     fan_line_rms=fan_line_rms, cv_raw=float(np.median([r["cv_raw"] for r in rows])),
                     cv_mult=cv_mult, speckle_var=speckle_v, line_cv_by_class=line_cv_cls, gain_lo=g_lo,
                     f0_mean={c: float(v.mean()) for c, v in f0s.items()}, f0_sigma_pct=f0_sig_pct,
                     f0_drift_pct_per_hour=drift, sameday_sigma_pct=sameday_sigma)
    cal.update(fan_power_3axis=fan_power, fan_line_rms=fan_line_rms, speed_sameday_sigma_pct=sameday_sigma)

    # ---------------------------------------------------------------- 7. mains hum
    hum = []
    for t in trials:
        fa, A = F.amp_spectrum(prep[t.name].x)
        Av = np.sqrt((A ** 2).sum(axis=1))
        f0 = feat[t.name]["f0"]
        for fm in (50.0, 100.0, 150.0):
            line = Av[np.abs(fa - fm) <= 0.3].max()
            msk = (np.abs(fa - fm) <= 5) & (np.abs(fa - fm) > 1)
            if t.cls != "off":
                for k in range(1, 6):
                    msk &= np.abs(fa - k * f0) > 1.5
            fl = np.median(Av[msk])
            hum.append(dict(cls=t.cls, fm=fm, snr_db=float(20 * np.log10(line / fl)), amp=float(line), floor=float(fl)))
    max_snr = max(h["snr_db"] for h in hum)
    amp_floor = float(np.median([h["floor"] for h in hum]))
    hum_amp_bound = max(2 * amp_floor, max(h["amp"] for h in hum if h["snr_db"] >= 6) if max_snr >= 6 else 0)
    hum_ratio = float(hum_amp_bound / np.sqrt(2) / fan_line_rms)
    ev["mains_hum"] = dict(max_line_snr_db=max_snr, detected=bool(max_snr >= 6), amp_bound=hum_amp_bound,
                           ratio_bound=hum_ratio)

    # ---------------------------------------------------------------- 8. transients
    peak_off = float(max(np.abs(prep[t.name].x).max() for t in off))
    peak_edge = float(max(np.abs(prep[t.name].x_full[:int(C.FS)]).max() for t in off))
    ev["transients"] = dict(max_abs_off_window=peak_off, max_abs_off_first_second=peak_edge,
                            max_crest_off=float(max(feat[t.name]["crest_pc1"] for t in off)))

    # ---------------------------------------------------------------- 9-11. geometry, timing, ADC
    gu = np.array([prep[t.name].gravity / np.linalg.norm(prep[t.name].gravity) for t in trials])
    gmu = gu.mean(axis=0) / np.linalg.norm(gu.mean(axis=0))
    ev["orientation"] = dict(max_dev_deg=float(np.degrees(np.arccos(np.clip(gu @ gmu, -1, 1))).max()))
    aud = [L.audit(t) for t in trials]
    n_drop = sum(r["dropped_samples"] for r in aud)
    ev["timing"] = dict(fs_fit_min=min(r["fs_fit"] for r in aud), fs_fit_max=max(r["fs_fit"] for r in aud),
                        ts_resid_us=float(np.median([r["ts_resid_us"] for r in aud])),
                        dropped=n_drop, drop_bound=(n_drop if n_drop else 3.0) / sum(r["n_samples"] for r in aud))
    ev["adc"] = dict(value_spacing=float(np.median([r["value_spacing"] for r in aud])),
                     reported_resolution=float(off[0].meta.get("resolution", np.nan)),
                     q_noise_ratio=float(np.median([r["value_spacing"] for r in aud]) / np.sqrt(12) / floor_sigma))

    # ---------------------------------------------------------------- 12. supply voltage -> speed
    # equilibrium motor torque K V^2 s = fan load c n^2  ->  dn/n = 2s/(1+s) dV/V
    kap = []
    for c, v in f0s.items():
        for slip in (1 - v.mean() / 50.0, 1 - (v.mean() / 3) / 25.0):   # 2-pole: rot=f0 ; 4-pole: rot=f0/3
            kap.append(2 * slip / (1 + slip))
    kap_lo, kap_hi = float(min(kap)), float(max(kap))
    ev["speed"] = dict(kappa_lo=kap_lo, kappa_hi=kap_hi, day_pct_lo=6 * kap_lo, day_pct_hi=6 * kap_hi,
                       sag15_pct_hi=15 * kap_hi)
    cal["speed_day_pct"] = 6 * kap_hi

    # ---------------------------------------------------------------- 13. datasheets -> levels
    ds = []
    for name, nd in DATASHEET_ND:
        add = max(nd ** 2 - nd_ug ** 2, 0.0) * (G0 * 1e-6) ** 2 * b_eff
        snr = 10 * np.log10(fan_power / (3 * add)) if add > 0 else np.inf
        ds.append(dict(part=name, nd_ug=nd, psd_mult=(nd / nd_ug) ** 2, awgn_snr_db=float(snr)))
    worst = max(ds, key=lambda d: d["nd_ug"])
    ev["datasheets"] = ds

    # ---------------------------------------------------------------- realistic bands (level units)
    bands = {
        "gaussian": [worst["awgn_snr_db"], np.inf],
        "johnson_nyquist": [15.0, 45.0],
        "jn_proofmass": [1.0, worst["psd_mult"]],
        "pink": [0.0, max(fc_pink, 0.2)],
        "brownian": [0.0, rw],
        "popcorn": [0.0, rtn_bound],
        "salt_pepper": [0.0, p_bound],
        "speckle": [0.0, speckle_v],
        "coupling": [max(g_lo, 0.05), 1.0],
        "quantize": [ADC_EXAMPLES[0][1], ADC_EXAMPLES[4][1]],
        "footsteps": [0.0, 0.05],
        "knocks": [0.0, 0.1],
        "mains_hum": [0.0, hum_ratio],
        "appliance": None,
        "orientation": [0.0, 5.0],
        "speed_shift": [0.0, 2 * sameday_sigma],
        "speed_shift_day": [0.0, 6 * kap_hi],
        "dropouts": [0.0, 0.0],
        "lower_fs": [200.0, 400.0],
        "lower_fs_aliased": None,
        "mix_same_day": [0.0, 1.0], "mix_other_day": [0.0, 1.0], "mix_other_phone": [0.0, 1.0],
    }
    cal["bands"] = bands
    cal["evidence"] = ev
    cal["phone_catalogue"] = dict(psd_mult=[max(d["psd_mult"], 1.0) for d in ds],
                                  lsb=[v for _, v in ADC_EXAMPLES[:5]], fs=[400.0, 200.0])

    table = _table(ev, bands, nd_ug)
    C.CAL_FILE.write_text(json.dumps(_jsonable(cal), indent=1))
    import pandas as pd
    pd.DataFrame(table).to_csv(C.TAB / "noise_calibration.csv", index=False)
    if figs:
        _figures(f, S_off, S_syn, floor_psd, nd_ug, f_lf, S_lf, lw, lp, z, rows, hum, ds)
    if verbose:
        print(f"  floor {nd_ug:.1f} ug/rtHz (sigma {floor_sigma*1e3:.2f} mm/s^2/axis), kurtosis {ev['floor']['kurt']:.2f}, "
              f"synthetic sigma ratio {ev['floor']['synthetic_sigma_ratio']:.3f}")
        print(f"  fan power SNR {ev['fan']['native_snr_db']:.1f} dB, line RMS {fan_line_rms*1e3:.2f} mm/s^2, "
              f"envelope CV {cv_mult:.3f}, same-day speed sigma {sameday_sigma:.2f} %")
    return load_cal()


def _fmt(b):
    if b is None:
        return "uncalibrated"
    lo, hi = b
    return f"{lo:.3g} .. {hi:.3g}"


def _table(ev, bands, nd_ug):
    E = ev
    ds = {d["part"]: d for d in E["datasheets"]}
    return [
        dict(family="gaussian", model="white Gaussian (AWGN), i.i.d. per axis", unit="SNR [dB]",
             realistic=f"{bands['gaussian'][0]:.1f} dB .. inf (this phone: no extra noise)", evidence_type="measured + datasheet",
             evidence=(f"measured floor {nd_ug:.0f} ug/rtHz, kurtosis {E['floor']['kurt']:.2f} (Gaussian = 3), "
                       f"axis correlation {E['floor']['mean_abs_axis_corr']:.2f}; worst phone part 400 ug/rtHz -> "
                       f"{bands['gaussian'][0]:.1f} dB")),
        dict(family="johnson_nyquist", model="thermal 4kT*w0/(mQ), white Gaussian, shaped", unit="deg C",
             realistic="15 .. 45 C", evidence_type="physics + assumption",
             evidence="noise power ~ absolute temperature (Gabrielson 1993); worst case: whole measured floor thermal"),
        dict(family="jn_proofmass", model="thermal noise of an r-times smaller proof mass", unit="PSD multiplier r",
             realistic=_fmt(bands["jn_proofmass"]), evidence_type="datasheet",
             evidence="; ".join(f"{d['part']}: r = {d['psd_mult']:.2f}" for d in E["datasheets"])),
        dict(family="pink", model="1/f flicker, PSD = floor at corner fc", unit="Hz",
             realistic=_fmt(bands["pink"]), evidence_type="measured",
             evidence=f"fit S_w + S_p/f to fan-off PSD 0.2-20 Hz: corner {E['pink']['fc']:.3g} Hz"),
        dict(family="brownian", model="bias random walk", unit="m/s^2/sqrt(s)",
             realistic=_fmt(bands["brownian"]), evidence_type="measured (upper bound)",
             evidence=f"gravity-vector drift between fan-off trials over {E['brownian']['span_min']:.0f} min "
                      f"(incl. tape creep): {E['brownian']['sigma_rw']:.2e}"),
        dict(family="popcorn", model="random telegraph steps, 5 switches/s", unit="m/s^2",
             realistic=_fmt(bands["popcorn"]), evidence_type="measured (bound)",
             evidence=f"no 50-ms block step > {E['popcorn']['max_step_over_sigma']:.1f} sigma in fan-off data; "
                      f"5-sigma detectability bound {E['popcorn']['bound']:.2e} m/s^2"),
        dict(family="salt_pepper", model="+-full-scale (78.45 m/s^2) sample replacement", unit="fraction",
             realistic=_fmt(bands["salt_pepper"]), evidence_type="measured (bound)",
             evidence=f"{E['salt_pepper']['spikes']} outliers in {E['salt_pepper']['samples']} samples -> "
                      f"rate < {E['salt_pepper']['rate_bound']:.1e} (rule of three, 95 %)"),
        dict(family="speckle", model="imnoise speckle J = I + n.*I, n uniform", unit="variance",
             realistic=_fmt(bands["speckle"]), evidence_type="measured",
             evidence=f"fan-line envelope wobble CV {E['fan']['cv_raw']:.3f} raw, {E['fan']['cv_mult']:.3f} after "
                      f"removing the additive-noise part; white speckle with the same +-3 Hz wobble"),
        dict(family="coupling", model="vibration gain g, sensor noise unchanged", unit="gain",
             realistic=_fmt(bands["coupling"]), evidence_type="measured (same mount)",
             evidence=f"trial-to-trial fan-line amplitude CV by class {E['fan']['line_cv_by_class']}; re-taping unknown"),
        dict(family="quantize", model="ADC rounding (gravity included)", unit="m/s^2",
             realistic=_fmt(bands["quantize"]), evidence_type="measured + datasheet",
             evidence=(f"this phone: value spacing {E['adc']['value_spacing']:.4f} (reported resolution "
                       f"{E['adc']['reported_resolution']}), quantisation noise {E['adc']['q_noise_ratio']:.2f} x floor; "
                       + ", ".join(f"{n}: {v:.4f}" for n, v in ADC_EXAMPLES))),
        dict(family="footsteps", model="impulse train 1.6-2.2 Hz, floor/wall modes 8-90 Hz", unit="peak m/s^2",
             realistic=_fmt(bands["footsteps"]), evidence_type="literature + measured bound",
             evidence=(f"floors designed for walking peaks < 0.5 %g = 0.049 m/s^2 (AISC Design Guide 11); "
                       f"largest event in 70 min of fan-off windows: {E['transients']['max_abs_off_window']:.3f} m/s^2")),
        dict(family="knocks", model="1-3 impacts, 10-180 Hz modes", unit="peak m/s^2",
             realistic=_fmt(bands["knocks"]), evidence_type="literature-anchored assumption",
             evidence="0.41 m/s^2 measured ON an impacted wood floor (Ding et al. 2020); a wall sensor in the same "
                      "room sees less -> 0.1 assumed"),
        dict(family="mains_hum", model="100 Hz + 50/150 Hz fixed tones", unit="RMS / fan-line RMS",
             realistic=_fmt(bands["mains_hum"]), evidence_type="measured (bound)",
             evidence=f"largest 50/100/150 Hz line {E['mains_hum']['max_line_snr_db']:.1f} dB above floor "
                      f"(detected: {E['mains_hum']['detected']}) -> amplitude < {E['mains_hum']['amp_bound']:.1e} m/s^2"),
        dict(family="appliance", model="47.5-49.5 Hz compressor + harmonics", unit="RMS / fan-line RMS",
             realistic="uncalibrated", evidence_type="uncalibrated",
             evidence="frequency from 2-pole motor physics; amplitude needs a recording with the appliance running"),
        dict(family="orientation", model="rigid rotation (gravity included)", unit="deg",
             realistic="0 .. 5", evidence_type="measured + assumption",
             evidence=f"in-session gravity-direction drift {E['orientation']['max_dev_deg']:.2f} deg; re-taping "
                      f"inside the marked outline assumed < 5 deg"),
        dict(family="speed_shift", model="band-limited time scaling of all fan lines", unit="+-%",
             realistic=(f"same day 0 .. {bands['speed_shift'][1]:.2f}; other day 0 .. {bands['speed_shift_day'][1]:.1f}"),
             evidence_type="measured + literature",
             evidence=(f"in-session f0 sigma {E['fan']['f0_sigma_pct']} %, drift {E['fan']['f0_drift_pct_per_hour']} %/h; "
                       f"supply tolerance +-6 % (Indian Electricity Rules / state supply codes) x dn/n = 2s/(1+s) dV/V "
                       f"= {E['speed']['kappa_lo']:.2f}..{E['speed']['kappa_hi']:.2f} -> +-{E['speed']['day_pct_lo']:.1f}.."
                       f"{E['speed']['day_pct_hi']:.1f} %; ESMI-type sags of -15 % -> up to {E['speed']['sag15_pct_hi']:.0f} %")),
        dict(family="dropouts", model="20-150 ms gaps", unit="% samples", realistic="0 (this phone)",
             evidence_type="measured",
             evidence=f"{E['timing']['dropped']} dropped samples; fs {E['timing']['fs_fit_min']:.3f}-"
                      f"{E['timing']['fs_fit_max']:.3f} Hz, timestamp scatter {E['timing']['ts_resid_us']:.2f} us; "
                      f"other phones show sampling instability (Stisen et al. 2015)"),
        dict(family="lower_fs", model="lower ODR with on-chip anti-alias LPF", unit="Hz",
             realistic="200 .. 400", evidence_type="literature",
             evidence="Android 12+ caps motion sensors at 200 Hz unless the app holds HIGH_SAMPLING_RATE_SENSORS"),
        dict(family="lower_fs_aliased", model="lower ODR without anti-alias filter", unit="Hz",
             realistic="n/a (modern MEMS filter on-chip)", evidence_type="stress test", evidence="worst case"),
    ]


def _figures(f, S_off, S_syn, floor_psd, nd_ug, f_lf, S_lf, lw, lp, z, rows, hum, ds):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    for k, lab in enumerate("xyz"):
        a.semilogy(f, S_off[:, k], lw=1, label=f"fan off, axis {lab}")
    for d in ds:
        a.axhline((d["nd_ug"] * 1e-6 * G0) ** 2, ls=":", lw=1, color="k")
        a.text(192, (d["nd_ug"] * 1e-6 * G0) ** 2, f"{d['nd_ug']:.0f}", fontsize=7, va="bottom", ha="right")
    a.axhline(floor_psd, color="tab:red", lw=1.5, label=f"measured plateau {nd_ug:.0f} ug/rtHz")
    a.set(xlabel="Hz", ylabel="PSD [(m/s^2)^2/Hz]", title="(a) real noise floor vs datasheet densities (dotted, ug/rtHz)")
    a.legend(fontsize=7)
    a = ax[0, 1]
    xx = np.linspace(-6, 6, 241)
    a.hist(z, bins=241, range=(-6, 6), density=True, color="0.6", label="fan-off samples (normalised)")
    a.semilogy(xx, stats.norm.pdf(xx), "r", label="Gaussian")
    a.set(ylim=(1e-6, 1), title=f"(b) amplitude distribution, kurtosis {stats.kurtosis(z, fisher=False):.2f}")
    a.legend(fontsize=7)
    a = ax[1, 0]
    a.loglog(f_lf[1:], S_lf[1:], "k", lw=1, label="fan-off, raw (no HP), detrended")
    ff = f_lf[1:]
    a.loglog(ff, np.exp(lw) + np.exp(lp) / ff, "r--", label=f"fit white + 1/f, corner {np.exp(lp) / np.exp(lw):.2g} Hz")
    a.set(xlabel="Hz", ylabel="PSD", title="(c) low-frequency floor: flicker (1/f) corner")
    a.legend(fontsize=7)
    a = ax[1, 1]
    a.semilogy(f, S_off.mean(axis=1), "k", label="real fan-off (mean of axes)")
    a.semilogy(f, S_syn.mean(axis=1), "r--", label="synthetic shaped Gaussian, same pipeline")
    a.set(xlabel="Hz", title="(d) validation: synthetic noise model vs real")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIG / "calib_noise_floor.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for c in C.CLASSES[1:]:
        rr = [r for r in rows if r["cls"] == c]
        t0 = min(r["epoch"] for r in rows)
        fv = np.array([r["f0"] for r in rr])
        ax[0].plot([(r["epoch"] - t0) / 60 for r in rr], (fv / fv.mean() - 1) * 100, "o-", color=C.CLASS_COLORS[c], label=c, ms=4)
        ax[1].plot([r["cv_raw"] for r in rr], [r["cv_mult"] for r in rr], "o", color=C.CLASS_COLORS[c], label=c, ms=4)
    ax[0].set(xlabel="minutes into session", ylabel="f0 deviation from class mean [%]", title="(a) fan speed stability (same day)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=.3)
    ax[1].set(xlabel="envelope CV (raw)", ylabel="CV after removing additive-noise part",
              title="(b) amplitude wobble of the dominant line")
    ax[1].grid(alpha=.3)
    for j, fm in enumerate((50, 100, 150)):
        for c in C.CLASSES:
            v = [h["snr_db"] for h in hum if h["cls"] == c and h["fm"] == fm]
            ax[2].plot(np.full(len(v), j) + 0.15 * C.CLASSES.index(c) - 0.2, v, "o", ms=3, color=C.CLASS_COLORS[c],
                       label=c if j == 0 else None)
    ax[2].axhline(6, color="k", ls=":", label="6 dB detection")
    ax[2].set(xticks=[0, 1, 2], xticklabels=["50 Hz", "100 Hz", "150 Hz"], ylabel="line SNR [dB]",
              title="(c) mains hum at the mount")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(C.FIG / "calib_fan_and_hum.png", dpi=110)
    plt.close(fig)
