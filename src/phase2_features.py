"""Phase 2 - features (Weeks 2-3 deliverable): feature table, discriminability ranking and the
FFT / CWT / time-series figures.

Outputs
  feature_table.csv                 one row per trial (all features incl. CWT dominant frequency)
  tables/feature_ranking.csv        Fisher ratio and worst-pair separation per feature
  tables/frequency_summary.csv      dominant frequency per class vs rotation hypotheses
  figures/p2_*.png
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from . import config as C
from . import features as F
from . import ml
from . import preprocess as P


def run(trials):
    prep = [P.preprocess(t.t, t.a) for t in trials]
    rows = []
    for t, p in zip(trials, prep):
        r = dict(trial=t.name, cls=t.cls, label=t.label, round=t.nn, start=t.meta.get("start_text", ""))
        r.update(F.extract(p, cwt=True))
        rows.append(r)
    df = pd.DataFrame(rows)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RESULTS / "feature_table.csv", index=False)

    names = F.FEATURE_NAMES + ["cwt_fdom"]
    X = df[names].to_numpy()
    y = df["label"].to_numpy()
    fr = ml.fisher_ratio(X, y)
    rank = []
    for j, n in enumerate(names):
        gaps = []
        for a, b in itertools.combinations(range(4), 2):
            xa, xb = X[y == a, j], X[y == b, j]
            lo, hi = (xa, xb) if xa.mean() < xb.mean() else (xb, xa)
            gaps.append((hi.min() - lo.max()) / np.sqrt(0.5 * (xa.var() + xb.var()) + 1e-30))
        row = dict(feature=n, fisher=fr[j], worst_pair_gap_sd=min(gaps),
                   separable_pairs=int(sum(g > 0 for g in gaps)))
        for c in range(4):
            row[f"mean_{C.CLASSES[c]}"] = X[y == c, j].mean()
            row[f"sd_{C.CLASSES[c]}"] = X[y == c, j].std()
        rank.append(row)
    rank = pd.DataFrame(rank).sort_values("fisher", ascending=False)
    rank.to_csv(C.TAB / "feature_ranking.csv", index=False)

    freq = []
    for c in C.CLASSES[1:]:
        d = df[df.cls == c]
        f0 = d["f0"]
        freq.append(dict(cls=c, fft_fdom_mean=d["fft_fdom_vec"].mean(), fft_fdom_sd=d["fft_fdom_vec"].std(),
                         cwt_fdom_mean=d["cwt_fdom"].mean(), f0_mean=f0.mean(), f0_sd=f0.std(),
                         rpm_if_rotation_is_f0=60 * f0.mean(), rpm_if_rotation_is_f0_over_3=20 * f0.mean(),
                         harmonics_present=f"{d['n_harm'].min()}-{d['n_harm'].max()} of 4"))
    freq = pd.DataFrame(freq)
    freq.to_csv(C.TAB / "frequency_summary.csv", index=False)

    rep = _representatives(df)
    _fig_time(trials, prep, rep)
    _fig_fft(trials, prep, rep, df)
    _fig_psd(prep, df)
    _fig_cwt(prep, rep, df)
    _fig_ranking(rank)
    _fig_scatter(df)
    env = _fig_envelope(prep, df)
    cwt_err = float(np.abs(df.loc[df.cls != "off", "cwt_fdom"] - df.loc[df.cls != "off", "fft_fdom_vec"]).max())
    return dict(top_features=rank.head(8)[["feature", "fisher", "worst_pair_gap_sd"]].to_dict("records"),
                freq=freq.to_dict("records"), representatives={C.CLASSES[c]: trials[i].name for c, i in rep.items()},
                cwt_vs_fft_max_diff_hz=cwt_err, envelope=env)


def _representatives(df):
    rep = {}
    for c in range(4):
        d = df[df.label == c]
        z = sum(((d[k] - d[k].median()) / (d[k].std() + 1e-12)) ** 2 for k in ("f0", "comb_snr", "rms_vec"))
        rep[c] = int(z.idxmin())
    return rep


def _fig_time(trials, prep, rep):
    fig, ax = plt.subplots(4, 2, figsize=(15, 10), gridspec_kw=dict(width_ratios=[2.2, 1]))
    for c, i in rep.items():
        x = prep[i].x
        tt = np.arange(len(x)) / C.FS
        for k, lab in enumerate("xyz"):
            ax[c, 0].plot(tt, x[:, k] * 1e3 + (1 - k) * 60, lw=0.4, label=lab if c == 0 else None)
            m = tt < 0.1
            ax[c, 1].plot(tt[m] * 1e3, x[m, k] * 1e3 + (1 - k) * 40, lw=1)
        ax[c, 0].set_ylabel(f"{C.CLASSES[c]}\n[mm/s^2]\n(offset)")
        ax[c, 0].set_title(f"{trials[i].name}: gravity-free acceleration, 5 s analysed window (x/y/z offset)", fontsize=9)
        ax[c, 1].set_title("first 100 ms", fontsize=9)
        ax[c, 0].set_ylim(-120, 120)
        ax[c, 1].set_ylim(-90, 90)
    ax[0, 0].legend(ncol=3, fontsize=8, loc="upper right")
    ax[3, 0].set_xlabel("time [s]")
    ax[3, 1].set_xlabel("time [ms]")
    fig.tight_layout()
    fig.savefig(C.FIG / "p2_time_series.png", dpi=110)
    plt.close(fig)


def _fig_fft(trials, prep, rep, df):
    fig, ax = plt.subplots(4, 1, figsize=(13, 11), sharex=True)
    for c, i in rep.items():
        f, A = F.amp_spectrum(prep[i].x)
        Av = np.sqrt((A ** 2).sum(axis=1))
        ax[c].semilogy(f, Av * 1e3, "k", lw=0.7)
        for m in (50, 100, 150):
            ax[c].axvspan(m - 1, m + 1, color="0.85", zorder=0)
        if c > 0:
            f0 = df.loc[i, "f0"]
            for k in range(1, 6):
                ax[c].axvline(k * f0, color="tab:red" if k == 3 else "tab:orange", ls="--", lw=0.8)
                ax[c].text(k * f0 + 0.8, Av.max() * 1e3 * 0.6, f"{k}f0" if k != 3 else "3f0 (dominant)", fontsize=8,
                           color="tab:red" if k == 3 else "tab:orange")
            ax[c].set_title(f"{trials[i].name}: f0 = {f0:.2f} Hz = {60 * f0:.0f} rpm if f0 is the shaft rate (envelope "
                            f"test supports this); dominant 3f0 = {3 * f0:.1f} Hz = blade-pass if 3 blades", fontsize=9)
        else:
            ax[c].set_title(f"{trials[i].name}: no tonal content (sensor noise floor)", fontsize=9)
        ax[c].set_ylabel("amplitude [mm/s^2]")
        ax[c].set_ylim(1e-2, 30)
        ax[c].grid(alpha=.3, which="both")
    ax[3].axvline(C.FS / 2, color="k", lw=1)
    ax[3].text(C.FS / 2 - 2, 10, "Nyquist", ha="right", fontsize=8)
    ax[3].set_xlabel("frequency [Hz]  (Hann window, 5 s, amplitude-corrected, vector of 3 axes; grey = mains bands)")
    ax[3].set_xlim(0, 202)
    fig.tight_layout()
    fig.savefig(C.FIG / "p2_fft_per_class.png", dpi=110)
    plt.close(fig)


def _fig_psd(prep, df):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.5), gridspec_kw=dict(width_ratios=[2, 1]))
    for i, p in enumerate(prep):
        f, S = F.welch_axes(p.x)
        c = df.loc[i, "cls"]
        for a in ax:
            a.semilogy(f, S.sum(axis=1), color=C.CLASS_COLORS[c], lw=0.6, alpha=0.6,
                       label=c if (a is ax[0] and df.loc[:i, "cls"].tolist().count(c) == 1) else None)
    ax[0].set(xlabel="Hz", ylabel="PSD, sum of axes [(m/s^2)^2/Hz]", title="Welch PSD of all 80 trials (Hann, 1024, 50 %)")
    ax[0].legend()
    ax[1].set(xlim=(85, 120), xlabel="Hz", title="zoom: dominant line 3f0")
    for a in ax:
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(C.FIG / "p2_psd_all.png", dpi=110)
    plt.close(fig)


def _fig_cwt(prep, rep, df):
    freqs = np.arange(20, 190, 0.5)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    for c, i in rep.items():
        s = F.pca_axis(prep[i].x)
        W = np.abs(F.cwt_morlet(s, C.FS, freqs)) * 1e3
        a = ax.flat[c]
        im = a.pcolormesh(np.arange(len(s)) / C.FS, freqs, 20 * np.log10(W + 1e-3), shading="auto",
                          cmap="magma", vmin=-10, vmax=25)
        if c > 0:
            a.axhline(3 * df.loc[i, "f0"], color="c", ls=":", lw=0.8)
        a.set_title(f"{C.CLASSES[c]} ({df.loc[i, 'trial']}): Morlet CWT of the principal axis", fontsize=9)
    for a in ax[:, 0]:
        a.set_ylabel("frequency [Hz]")
    for a in ax[1]:
        a.set_xlabel("time [s]")
    fig.colorbar(im, ax=ax, label="|W| [dB re 1 mm/s^2]")
    fig.savefig(C.FIG / "p2_cwt_per_class.png", dpi=110)
    plt.close(fig)


def _fig_ranking(rank):
    fig, ax = plt.subplots(figsize=(12, 5))
    r = rank.reset_index(drop=True)
    ax.bar(range(len(r)), r["fisher"], color=["tab:green" if g > 0 else "tab:gray" for g in r["worst_pair_gap_sd"]])
    ax.set_yscale("log")
    ax.set_xticks(range(len(r)))
    ax.set_xticklabels(r["feature"], rotation=70, fontsize=8)
    ax.set(ylabel="Fisher ratio (between / within class variance)",
           title="Feature discriminability on clean data (green: all 6 class pairs fully separated)")
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(C.FIG / "p2_feature_ranking.png", dpi=110)
    plt.close(fig)


def _fig_scatter(df):
    pairs = [("rms_z", "fft_fdom_z", "textbook pair"), ("comb_snr", "f0", "physics pair (used by T2)"),
             ("kurt_pc1", "spec_flatness", "shape statistics")]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (fx, fy, title) in zip(ax, pairs):
        for c in C.CLASSES:
            d = df[df.cls == c]
            a.scatter(d[fx], d[fy], s=18, color=C.CLASS_COLORS[c], label=c)
        a.set(xlabel=fx, ylabel=fy, title=title)
        a.grid(alpha=.3)
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(C.FIG / "p2_feature_scatter.png", dpi=110)
    plt.close(fig)


def _fig_envelope(prep, df):
    """Rotation test: if the shaft turned at f0/3 (~11-12 Hz), the dominant line would carry
    once-per-revolution modulation -> envelope spectrum peak at f0/3 and spectral lines at
    multiples of f0/3. If the shaft turns at f0, neither appears."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    res = {}
    for c in C.CLASSES[1:]:
        idx = df.index[df.cls == c]
        Es, Ss = [], []
        for i in idx:
            f0 = df.loc[i, "f0"]
            s = F.pca_axis(prep[i].x)
            sos = signal.butter(4, [3 * f0 - 15, 3 * f0 + 15], "bandpass", fs=C.FS, output="sos")
            e = np.abs(signal.hilbert(signal.sosfiltfilt(sos, s)))
            fe, E = signal.welch(e - e.mean(), fs=C.FS, nperseg=1024)
            Es.append(E / np.median(E[(fe > 2) & (fe < 30)]))
            fs_, S = signal.welch(s, fs=C.FS, nperseg=1024)
            Ss.append(S / np.median(S[(fs_ > 2) & (fs_ < 30)]))
            # peak prominence at f0/3: value / local median (+-4 Hz, excluding +-0.6 Hz)
            loc = (np.abs(fe - f0 / 3) <= 4) & (np.abs(fe - f0 / 3) > 0.6)
            res.setdefault(c, []).append(float(10 * np.log10(np.interp(f0 / 3, fe, Es[-1]) / np.median(Es[-1][loc]))))
        ax[0].plot(fe, 10 * np.log10(np.mean(Es, axis=0)), color=C.CLASS_COLORS[c], label=c)
        ax[1].plot(fs_, 10 * np.log10(np.mean(Ss, axis=0)), color=C.CLASS_COLORS[c], label=c)
        f0m = df.loc[idx, "f0"].mean()
        for a in ax:
            a.axvline(f0m / 3, color=C.CLASS_COLORS[c], ls=":", lw=0.8)
    ax[0].set(xlim=(0, 30), ylim=(-30, 20), xlabel="modulation frequency [Hz]", ylabel="dB re median",
              title="(a) envelope spectrum of the 3f0 +-15 Hz band (dotted: f0/3)")
    ax[1].set(xlim=(0, 45), xlabel="Hz", ylabel="dB re median",
              title="(b) low-frequency spectrum: any line at f0/3 or 2f0/3?")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(C.FIG / "p2_envelope_rotation_test.png", dpi=110)
    plt.close(fig)
    return {c: f"{np.median(v):+.1f} dB prominence at f0/3" for c, v in res.items()}
