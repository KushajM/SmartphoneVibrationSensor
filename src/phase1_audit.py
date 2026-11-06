"""Phase 1 - acquisition audit (Week-1 deliverable) and preprocessing justification.

Outputs
  tables/audit.csv                 per-trial timing, gravity, quantisation, flags
  tables/edge_check.csv            RMS of first/last second vs centre (spin-up/handling check)
  figures/p1_audit.png             sampling rate, timestamp scatter, gravity drift over the session
  figures/p1_resampling.png        linear vs band-limited resampling (the "decay" artefact)
  figures/p1_gravity_removal.png   high-pass design and its effect on the spectrum
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from . import config as C
from . import features as F
from . import load_trials as L
from . import preprocess as P


def run(trials):
    C.FIG.mkdir(parents=True, exist_ok=True)
    C.TAB.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([L.audit(t) for t in trials]).sort_values("start_epoch").reset_index(drop=True)
    df.to_csv(C.TAB / "audit.csv", index=False)
    order = "".join("O123"[C.CLASSES.index(c)] for c in df["cls"])
    rr = all(order[i:i + 4] == "O123" for i in range(0, len(order), 4))
    span = (df["start_epoch"].max() - df["start_epoch"].min()) / 60
    gu = df[["g_x", "g_y", "g_z"]].to_numpy()
    gu = gu / np.linalg.norm(gu, axis=1, keepdims=True)
    gm = gu.mean(axis=0) / np.linalg.norm(gu.mean(axis=0))
    m0 = trials[0].meta
    summary = dict(
        n_trials=len(df), per_class=df["cls"].value_counts().to_dict(),
        device=m0.get("device", ""), sensor=m0.get("sensor", ""), range=m0.get("range"),
        reported_resolution=m0.get("resolution"), min_delay_us=m0.get("min_delay_us"),
        fs_fit_mean=df["fs_fit"].mean(), fs_fit_min=df["fs_fit"].min(), fs_fit_max=df["fs_fit"].max(),
        ts_resid_us_median=df["ts_resid_us"].median(), ts_resid_us_max=df["ts_resid_us"].max(),
        dt_std_us_median=df["dt_std_us"].median(), max_dt_ms=df["max_dt_ms"].max(),
        dropped_total=int(df["dropped_samples"].sum()),
        duration_min=df["duration_s"].min(), duration_max=df["duration_s"].max(),
        g_norm_mean=df["g_norm"].mean(), g_norm_sd=df["g_norm"].std(),
        tilt_inplane_mean=df["tilt_inplane_deg"].mean(), tilt_outplane_mean=df["tilt_outplane_deg"].mean(),
        orientation_drift_deg=float(np.degrees(np.arccos(np.clip(gu @ gm, -1, 1))).max()),
        value_spacing=df["value_spacing"].median(), session_minutes=span, round_robin=rr,
        flagged=df.loc[df["flag"] != "", "trial"].tolist())

    # ---- edge check: does the first/last second differ from the analysed centre?
    edge = []
    for t in trials:
        p = P.preprocess(t.t, t.a)
        x, n = p.x_full, int(C.FS)
        r = lambda s: float(np.sqrt(np.mean(s ** 2)))
        centre = r(p.x)
        edge.append(dict(trial=t.name, cls=t.cls, first_s=r(x[:n]) / centre, last_s=r(x[-n:]) / centre))
    edge = pd.DataFrame(edge)
    edge.to_csv(C.TAB / "edge_check.csv", index=False)
    summary["edge_first_median"] = edge["first_s"].median()
    summary["edge_last_median"] = edge["last_s"].median()
    summary["edge_max"] = float(edge[["first_s", "last_s"]].to_numpy().max())

    _fig_audit(df)
    summary.update(_fig_resampling(trials))
    _fig_gravity(trials)
    return summary


def _fig_audit(df):
    t0 = df["start_epoch"].min()
    mins = (df["start_epoch"] - t0) / 60
    col = [C.CLASS_COLORS[c] for c in df["cls"]]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].scatter(mins, df["fs_fit"], c=col, s=14)
    ax[0].set(xlabel="minutes into session", ylabel="fitted sampling rate [Hz]",
              title=f"(a) device rate {df['fs_fit'].mean():.3f} Hz, no dropped samples")
    ax[1].scatter(mins, df["ts_resid_us"], c=col, s=14)
    ax[1].set(xlabel="minutes into session", ylabel="timestamp scatter about clock fit [us]",
              title="(b) timestamp jitter (vs 2500 us sample period)")
    ax[2].plot(mins, (df["g_x"] - df["g_x"].iloc[0]) * 1e3, "o-", ms=3, label="g_x")
    ax[2].plot(mins, (df["g_z"] - df["g_z"].iloc[0]) * 1e3, "s-", ms=3, label="g_z")
    ax[2].plot(mins, (df["g_y"] - df["g_y"].iloc[0]) * 1e3, "^-", ms=3, label="g_y")
    ax[2].set(xlabel="minutes into session", ylabel="change in mean acceleration [mm/s^2]",
              title="(c) gravity vector drift (mount creep)")
    ax[2].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(C.FIG / "p1_audit.png", dpi=110)
    plt.close(fig)


def _fig_resampling(trials):
    """Why linear interpolation was rejected: tone amplitude over the record."""
    on = [t for t in trials if t.cls != "off"]
    seg = 0.25
    prof = {"linear": [], "sinc": []}
    fs_ins = []
    for t in on:
        pp = {m: P.preprocess(t.t, t.a, method=m) for m in prof}
        fs_ins.append(pp["sinc"].fs_in)
        f0 = F.extract(pp["sinc"])["f0"]
        sos = signal.butter(4, [3 * f0 - 3, 3 * f0 + 3], "bandpass", fs=C.FS, output="sos")
        ref = None
        for m, p in pp.items():
            y = signal.sosfiltfilt(sos, p.x_full[:, 2])
            n = int(seg * C.FS)
            r = np.array([np.sqrt(np.mean(y[i:i + n] ** 2)) for i in range(0, len(y) - n + 1, n)])
            prof[m].append(r)
    L_ = min(len(v) for v in prof["sinc"])
    ref = np.mean([v[:L_] for v in prof["sinc"]], axis=0)
    tc = (np.arange(L_) + 0.5) * seg
    lin = np.mean([v[:L_] for v in prof["linear"]], axis=0) / ref
    snc = np.mean([v[:L_] for v in prof["sinc"]], axis=0) / ref
    fs_in = float(np.mean(fs_ins))
    mu = np.mod(tc * (fs_in - C.FS), 1.0)               # fractional offset between the two grids
    w = 2 * np.pi * 103.5 / fs_in
    theory = np.abs(1 - mu + mu * np.exp(-1j * w))
    i0 = (len(tc) * seg - C.TRIM_S) / 2                # start of the analysed (central) window
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(tc, lin, "o-", color="tab:red", label="linear interpolation (rejected)")
    ax.plot(tc, snc, "s-", color="tab:blue", label="Kaiser-windowed sinc (used)")
    ax.plot(tc, theory, "k--", label=f"theory |1-mu+mu e^(-jw)| at 103.5 Hz, mu = frac(t*{fs_in - C.FS:.3f} Hz)")
    ax.axvspan(i0, i0 + C.TRIM_S, color="0.9", zorder=0, label="analysed window")
    ax.set(xlabel="time in recording [s]", ylabel="dominant-line amplitude (rel. to sinc)", ylim=(0.5, 1.1),
           title="Resampling 400.108 Hz -> 400 Hz: linear interpolation fakes a start-locked amplitude decay")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(C.FIG / "p1_resampling.png", dpi=110)
    plt.close(fig)
    win = (tc >= i0) & (tc <= i0 + C.TRIM_S)
    return dict(linear_amp_in_window=float(lin[win].mean()), linear_amp_min=float(lin.min()),
                sinc_amp_spread=float(snc.max() - snc.min()))


def _fig_gravity(trials):
    t = next(t for t in trials if t.cls == "speed2")
    xs, fs_in, _ = P.regularise(t.t, t.a)
    y, _ = P.resample_uniform(xs, fs_in)
    f = np.linspace(0.01, 200, 4000)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].semilogx(f, 20 * np.log10(np.maximum(P.hp_response(f), 1e-12)))
    ax[0].axvline(C.HP_FC, color="k", ls=":")
    ax[0].axvspan(32, 147, color="tab:orange", alpha=.15, label="fan lines (32-147 Hz)")
    ax[0].set(xlim=(0.05, 200), ylim=(-120, 3), xlabel="Hz", ylabel="gain [dB]",
              title=f"(a) {C.HP_ORDER}th-order Butterworth HP {C.HP_FC} Hz, forward-backward (zero phase)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3, which="both")
    for lab, sig, st in [("raw (gravity in)", y - 0.0, "0.5"), ("mean subtracted", y - y.mean(0), "tab:green"),
                         ("high-passed (used)", P.highpass(y), "tab:blue")]:
        fw, S = signal.welch(sig[:, 1], fs=C.FS, nperseg=2048)
        ax[1].loglog(fw[1:], S[1:], color=st, label=lab, lw=1)
    ax[1].set(xlabel="Hz", ylabel="PSD, y axis (gravity axis)", title="(b) effect on the spectrum (speed2 trial)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(C.FIG / "p1_gravity_removal.png", dpi=110)
    plt.close(fig)
