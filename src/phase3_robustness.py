"""Phase 3b - robustness of every model to the calibrated noise families.

Protocol
  * repeated stratified 5-fold CV by TRIAL (R repeats). Models are trained on the training
    folds only (clean models: real trials; *_aug models: real trials + noise-augmented copies).
  * every TEST trial is corrupted with a fresh random draw of the family at each level and
    pushed through the complete pipeline.
  * leave-one-group-out augmentation: when family X is tested, every family of X's group is
    excluded from the augmentation bank -> robustness is measured on UNSEEN noise types.
    (The three calibrated mixes are in-distribution for the augmented models; flagged.)
  * realistic levels (from calibrate_noise.py) are shaded in the curves; everything beyond is
    a stress test.
Ablation: each robust front-end component is switched off on the noise it was designed for.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import delayed

from . import config as C
from . import ml
from . import noise_models as N
from . import phase3_classify as P3
from . import workers as W
from .calibrate_noise import load_cal

STYLE = {"T1_textbook": ("0.55", ":"), "T2_physics": ("k", "-"), "kNN_clean": ("tab:green", "--"),
         "kNN_aug": ("tab:green", "-"), "LR_clean": ("tab:blue", "--"), "LR_aug": ("tab:blue", "-"),
         "HLR_clean": ("tab:red", "--"), "HLR_aug": ("tab:red", "-"), "HLR_aug_known": ("darkred", "-.")}
EVAL_MODELS = ml.MODELS + ["HLR_aug_known"]   # *_aug = noise type UNSEEN in training (strict);
                                              # HLR_aug_known = final model, all noise types in training
ABLATION = [("Hampel despiking", dict(despike=False), "salt_pepper", 1e-3),
            ("median-averaged Welch", dict(opts={"median": False}), "knocks", 1.0),
            ("mains-band mask", dict(opts={"mains_mask": False}), "mains_hum", 4),
            ("f0 prior 28-42 Hz", dict(opts={"f0_prior": False}), "gaussian", -10),
            ("3-axis PSD sum", dict(opts={"axis_sum": False}), "orientation", 90)]


def in_band(fam, lv, cal, key=None):
    b = cal["bands"].get(key or fam)
    if b is None:
        return lv in (N.SPEC[fam]["levels"][0],)
    lo, hi = b
    return bool(lo - 1e-12 <= lv <= hi + 1e-12)


def _lvl(v):
    return "clean" if (isinstance(v, float) and np.isinf(v)) else f"{v:g}"


def run(R=5, K=4, families=None, verbose=True):
    cal = load_cal()
    df, X, y, _ = P3.load_table()
    n = len(y)
    fams = families or list(N.SPEC)
    groups = sorted({N.SPEC[f]["group"] for f in fams} - {"mix"})
    banks = {"all": P3.bank_cache("all", n, K)}
    for g in groups:
        banks[g] = P3.bank_cache(g, n, K)

    keys = [(f, lv, r) for f in fams for lv in N.SPEC[f]["levels"] for r in range(R)]
    feats = dict(zip(keys, W.run([delayed(W.corrupt_task)(f, lv, r) for f, lv, r in keys])))
    if verbose:
        print(f"  corrupted feature sets: {len(keys)} x {n} trials")

    folds = {r: ml.stratified_folds(y, 5, seed=100 + r) for r in range(R)}
    clean_m = [m for m in ml.MODELS if not m.endswith("_aug")]
    aug_m = [m for m in ml.MODELS if m.endswith("_aug")]
    fkeys = ([("clean", m, r, k) for r in range(R) for k in range(5) for m in clean_m]
             + [(g, m, r, k) for g in ["all"] + groups for r in range(R) for k in range(5) for m in aug_m])
    fitted = dict(zip(fkeys, W.run([delayed(P3._fit)(m, X, y, folds[r][k][0], None if g == "clean" else banks[g],
                                                     100 * r + k) for g, m, r, k in fkeys])))
    if verbose:
        print(f"  fitted models: {len(fkeys)}")

    rows = []
    for f in fams:
        g = N.SPEC[f]["group"]
        gk = "all" if g == "mix" else g
        for lv in N.SPEC[f]["levels"]:
            for m in EVAL_MODELS:
                accs = []
                for r in range(R):
                    ok = 0
                    for k, (tr, te) in enumerate(folds[r]):
                        if m == "HLR_aug_known":
                            mod = fitted[("all", "HLR_aug", r, k)]
                        else:
                            mod = fitted[("clean", m, r, k)] if m in clean_m else fitted[(gk, m, r, k)]
                        ok += int((mod.predict(feats[(f, lv, r)][te]) == y[te]).sum())
                    accs.append(ok / n)
                rows.append(dict(family=f, group=g, level=lv, model=m, acc=float(np.mean(accs)),
                                 sd=float(np.std(accs)), realistic=in_band(f, lv, cal),
                                 realistic_other_day=in_band(f, lv, cal, "speed_shift_day") if f == "speed_shift" else None))
    res = pd.DataFrame(rows)
    res.to_csv(C.TAB / "robustness.csv", index=False)

    summ = []
    for (f, m), d in res.groupby(["family", "model"], sort=False):
        d = d.reset_index(drop=True)
        real = d[d.realistic]
        below = d.index[d.acc < 0.9]
        calibrated = cal["bands"].get(f) is not None
        if len(real) and calibrated:
            iw = real.acc.idxmin()
            wr, wr_sd, wr_lv = real.acc[iw], real.sd[iw], _lvl(real.level[iw])
            rm, rs = real.acc.mean(), real.sd.mean()
        else:
            wr = wr_sd = rm = rs = np.nan
            wr_lv = ""
        summ.append(dict(family=f, model=m,
                         worst_realistic=wr, worst_realistic_sd=wr_sd, worst_realistic_level=wr_lv,
                         mean_realistic=rm, mean_realistic_sd=rs,
                         harshest_level=_lvl(d.level.iloc[-1]), acc_harshest=d.acc.iloc[-1],
                         acc_harshest_sd=d.sd.iloc[-1],
                         first_level_below_90=_lvl(d.level[below[0]]) if len(below) else "none"))
    summ = pd.DataFrame(summ)
    summ.to_csv(C.TAB / "robustness_summary.csv", index=False)

    abl = _ablation(X, y, feats, folds, R, banks)
    _fig_curves(res, cal, fams)
    _fig_heatmap(summ, fams)
    _fig_ablation(abl)
    return dict(summary=summ, ablation=abl, res=res)


def _ablation(X, y, feats, folds, R, banks):
    out = []
    n = len(y)
    for name, kw, fam, lv in ABLATION:
        Xa = W.clean_features(n, **kw)
        Fa = W.run([delayed(W.corrupt_task)(fam, lv, r, **kw) for r in range(R)])
        for variant, Xc, Fs in (("full front-end", X, [feats[(fam, lv, r)] for r in range(R)]),
                                (f"without {name}", Xa, Fa)):
            for m in ("T2_physics", "HLR_clean"):
                accs = []
                for r in range(R):
                    ok = 0
                    for k, (tr, te) in enumerate(folds[r]):
                        mod = ml.fit_model(m, Xc, y, tr, None, seed=k)
                        ok += int((mod.predict(Fs[r][te]) == y[te]).sum())
                    accs.append(ok / n)
                out.append(dict(component=name, test=f"{fam} @ {lv:g}", variant=variant, model=m,
                                acc=float(np.mean(accs)), sd=float(np.std(accs))))
    out = pd.DataFrame(out)
    out.to_csv(C.TAB / "ablation.csv", index=False)
    return out


def _fig_curves(res, cal, fams):
    ncol = 5
    nrow = int(np.ceil(len(fams) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.3 * nrow))
    for ax, f in zip(axs.flat, fams):
        d = res[res.family == f]
        lv = N.SPEC[f]["levels"]
        xi = np.arange(len(lv))
        real = [i for i, v in enumerate(lv) if in_band(f, v, cal)]
        if real:
            ax.axvspan(min(real) - 0.4, max(real) + 0.4, color="tab:green", alpha=0.10, lw=0)
        if f == "speed_shift":
            day = [i for i, v in enumerate(lv) if in_band(f, v, cal, "speed_shift_day")]
            ax.axvspan(max(real) + 0.4, max(day) + 0.4, color="gold", alpha=0.15, lw=0)
        for m in EVAL_MODELS:
            dm = d[d.model == m]
            c, ls = STYLE[m]
            mu, sd = dm.acc.to_numpy() * 100, dm.sd.to_numpy() * 100
            ax.plot(xi, mu, ls, color=c, marker=".", lw=1.6 if m == "HLR_aug" else 1.0, label=m)
            ax.fill_between(xi, mu - sd, mu + sd, color=c, alpha=0.10, lw=0)          # +-1 sd over repeats
        ax.set_xticks(xi)
        ax.set_xticklabels([_lvl(v) for v in lv], fontsize=7, rotation=30)
        ax.set_ylim(0, 103)
        ax.axhline(25, color="k", lw=0.5, ls=":")
        tag = " (in-distribution)" if N.SPEC[f]["group"] == "mix" else ""
        ax.set_title(f + tag, fontsize=9)
        ax.set_xlabel(N.SPEC[f]["unit"], fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axs.flat[len(fams):]:
        ax.axis("off")
    axs.flat[0].legend(fontsize=6, loc="lower left")
    fig.suptitle("Accuracy (mean +- 1 sd over 5 repeated 5-fold CVs, shaded) vs noise level; test-time corruption, "
                 "leave-one-group-out augmentation. Green = calibrated realistic range, gold = other-day voltage range",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(C.FIG / "p3_robustness_curves.png", dpi=95)
    plt.close(fig)


def _fig_heatmap(summ, fams):
    fig, ax = plt.subplots(1, 2, figsize=(16, 0.42 * len(fams) + 2))
    for a, col, title in ((ax[0], "worst_realistic", "worst accuracy within the calibrated realistic range (mean +- sd)"),
                          (ax[1], "acc_harshest", "accuracy at the harshest stress level tested (mean +- sd)")):
        M = summ.pivot(index="family", columns="model", values=col).loc[fams, EVAL_MODELS].to_numpy() * 100
        Sd = summ.pivot(index="family", columns="model", values=col.replace("acc_harshest", "acc_harshest_sd")
                        if col == "acc_harshest" else "worst_realistic_sd").loc[fams, EVAL_MODELS].to_numpy() * 100
        im = a.imshow(M, cmap="RdYlGn", vmin=25, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    a.text(j, i, f"{M[i, j]:.0f}\n+-{Sd[i, j]:.0f}", ha="center", va="center", fontsize=6)
        a.set_xticks(range(len(EVAL_MODELS)))
        a.set_xticklabels(EVAL_MODELS, rotation=35, fontsize=8)
        a.set_yticks(range(len(fams)))
        a.set_yticklabels(fams, fontsize=8)
        a.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="accuracy [%]", shrink=0.6)
    fig.savefig(C.FIG / "p3_robustness_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _fig_ablation(abl):
    fig, ax = plt.subplots(figsize=(11, 4))
    comps = abl.component.unique()
    for j, m in enumerate(("T2_physics", "HLR_clean")):
        d = abl[abl.model == m]
        full = [d[(d.component == c) & (d.variant == "full front-end")].acc.iloc[0] * 100 for c in comps]
        wo = [d[(d.component == c) & (d.variant != "full front-end")].acc.iloc[0] * 100 for c in comps]
        x = np.arange(len(comps)) + (j - 0.5) * 0.4
        ax.bar(x - 0.09, full, 0.18, color=STYLE[m][0], label=f"{m}: full front-end")
        ax.bar(x + 0.09, wo, 0.18, color=STYLE[m][0], alpha=0.35, label=f"{m}: component removed")
    ax.set_xticks(range(len(comps)))
    ax.set_xticklabels([f"{c}\n({t})" for c, t in abl.groupby("component", sort=False).test.first().items()], fontsize=8)
    ax.set(ylabel="accuracy [%]", ylim=(0, 105), title="Ablation: what each robust front-end component buys")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(C.FIG / "p3_ablation.png", dpi=110)
    plt.close(fig)
