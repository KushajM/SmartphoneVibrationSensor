"""Phase 3a - classification and validation on the real (clean) recordings.

  * Leave-one-round-out CV (20 folds; each fold holds out one full round-robin cycle = one
    trial of every class, recorded within ~4 minutes of each other)
  * repeated stratified 5-fold CV (10 repeats) for the spread of the estimate
  * Clopper-Pearson 95 % intervals, pooled confusion matrices, selected features per fold
  * decision margins, analysis-window-length study, final model (+ JSON for predict.py)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import delayed

from . import config as C
from . import features as F
from . import ml
from . import workers as W

K_AUG = 4


def _fit(name, X, y, tr, bank, seed):
    return ml.fit_model(name, X, y, tr, bank, seed)


def load_table():
    df = pd.read_csv(C.RESULTS / "feature_table.csv")
    return df, df[F.FEATURE_NAMES].to_numpy(), df["label"].to_numpy(), df["round"].to_numpy()


def bank_cache(key, n, K=K_AUG):
    path = C.RESULTS / "cache" / f"bank_{key}_K{K}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        z = np.load(path)
        return z["feats"], z["masks"]
    fb, mb = W.augmentation_bank(n, None if key == "all" else key, K)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, feats=fb, masks=mb)
    tmp.replace(path)                                   # atomic: readers never see a partial file
    return fb, mb


def cv(models, X, y, folds, bank, seed=0):
    jobs = [(m, k) for k in range(len(folds)) for m in models]
    fitted = W.run([delayed(_fit)(m, X, y, folds[k][0], bank, seed + k) for m, k in jobs])
    pred = {m: np.full(len(y), -1) for m in models}
    proba = np.zeros((len(y), 4))
    feats = {m: [] for m in models}
    thr = []
    for (m, k), mod in zip(jobs, fitted):
        te = folds[k][1]
        pred[m][te] = mod.predict(X[te])
        feats[m].append(tuple(getattr(mod, "features", [])))
        if m == "HLR_aug":
            proba[te] = mod.predict_proba(X[te])
        if m == "T2_physics":
            a, s = mod.margins(X[te])
            thr.append(pd.DataFrame(dict(idx=te, snr_margin_db=a, f0_margin_hz=s)))
    return pred, proba, feats, pd.concat(thr) if thr else None


def run(trials=None):
    df, X, y, rounds = load_table()
    n = len(y)
    bank = bank_cache("all", n)

    # ---------------------------------------------------------------- leave-one-round-out
    loro = ml.loro_folds(rounds)
    pred, proba, feats, marg = cv(ml.MODELS, X, y, loro, bank)
    rows, cms = [], {}
    for m in ml.MODELS:
        k = int((pred[m] == y).sum())
        lo, hi = ml.clopper_pearson(k, n)
        cms[m] = ml.confusion(y, pred[m])
        per_round = [float((pred[m][te] == y[te]).mean()) for _, te in loro]
        rows.append(dict(model=m, description=ml.MODEL_LABEL[m], loro_correct=k, loro_n=n, loro_acc=k / n,
                         ci95_lo=lo, ci95_hi=hi, loro_round_mean=float(np.mean(per_round)),
                         loro_round_sd=float(np.std(per_round))))
    # ---------------------------------------------------------------- repeated stratified 5-fold
    rep = {m: [] for m in ml.MODELS}
    for r in range(10):
        folds = ml.stratified_folds(y, 5, seed=1000 + r)
        p, _, _, _ = cv(ml.MODELS, X, y, folds, bank, seed=1000 + r)
        for m in ml.MODELS:
            rep[m].append(float((p[m] == y).mean()))
    for row in rows:
        row["rep5x10_mean"] = float(np.mean(rep[row["model"]]))
        row["rep5x10_sd"] = float(np.std(rep[row["model"]]))
    res = pd.DataFrame(rows)
    res.to_csv(C.TAB / "clean_cv.csv", index=False)

    sel = []
    for m in ml.MODELS:
        vc = pd.Series([" + ".join(f) for f in feats[m]]).value_counts()
        for fs, cnt in vc.items():
            sel.append(dict(model=m, features=fs, folds=int(cnt), of=len(loro)))
    sel = pd.DataFrame(sel)
    sel.to_csv(C.TAB / "selected_features.csv", index=False)

    # ---------------------------------------------------------------- margins
    marg = marg.groupby("idx").mean()
    on = y[marg.index] > 0
    margins = dict(T2_min_snr_margin_off_db=float(-marg["snr_margin_db"][~on].max()),
                   T2_min_snr_margin_on_db=float(marg["snr_margin_db"][on].min()),
                   T2_min_f0_margin_hz=float(marg["f0_margin_hz"][on].min()),
                   HLR_aug_min_true_class_prob=float(proba[np.arange(n), y].min()),
                   HLR_aug_median_true_class_prob=float(np.median(proba[np.arange(n), y])))

    # ---------------------------------------------------------------- window length
    win = []
    wmods = ["T2_physics", "HLR_clean", "LR_clean"]
    for T in (5.0, 2.0, 1.0, 0.5):
        Xw = X if T == C.TRIM_S else W.clean_features(n, trim=T)
        p, _, _, _ = cv(wmods, Xw, y, loro, None)
        reps = {m: [] for m in wmods}
        for r in range(10):
            pr, _, _, _ = cv(wmods, Xw, y, ml.stratified_folds(y, 5, seed=2000 + r), None, seed=2000 + r)
            for m in wmods:
                reps[m].append(float((pr[m] == y).mean()))
        for m in wmods:
            k = int((p[m] == y).sum())
            lo, hi = ml.clopper_pearson(k, n)
            win.append(dict(window_s=T, model=m, acc=k / n, ci95_lo=lo, ci95_hi=hi,
                            rep_mean=float(np.mean(reps[m])), rep_sd=float(np.std(reps[m])),
                            f0_sd_speed2=float(Xw[y == 2, F.FEATURE_NAMES.index("f0")].std())))
    win = pd.DataFrame(win)
    win.to_csv(C.TAB / "window_length.csv", index=False)

    # ---------------------------------------------------------------- final model
    final = ml.fit_model("HLR_aug", X, y, np.arange(n), bank, seed=7)
    t2 = ml.fit_model("T2_physics", X, y, np.arange(n))
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def stage(est, feats):
        sc, lr = est[0], est[1]
        return dict(features=feats, scaler_mean=sc.mean_.tolist(), scaler_scale=sc.scale_.tolist(),
                    coef=lr.coef_.tolist(), intercept=lr.intercept_.tolist(), classes=lr.classes_.tolist())
    model_json = dict(model="two-stage logistic regression (detect fan, then estimate speed), "
                            "trained on real + calibrated noise-augmented trials", classes=C.CLASSES,
                      stage_a=stage(final.A, final.features_a), stage_b=stage(final.B, final.features_b),
                      preprocessing=dict(fs=C.FS, trim_s=C.TRIM_S, hp_order=C.HP_ORDER, hp_fc=C.HP_FC),
                      pipeline=dict(fs=C.FS, trim_s=C.TRIM_S, sinc_half=C.SINC_HALF, sinc_beta=C.SINC_BETA,
                                    hp_order=C.HP_ORDER, hp_fc=C.HP_FC, hampel_half=C.HAMPEL_HALF,
                                    hampel_nsig=C.HAMPEL_NSIG, rob_nperseg=C.ROB_NPERSEG,
                                    floor_halfwidth_hz=C.FLOOR_HALFWIDTH_HZ, f0_min=C.F0_RANGE[0],
                                    f0_max=C.F0_RANGE[1], f0_step=C.F0_STEP,
                                    f0_n=int(len(np.arange(C.F0_RANGE[0], C.F0_RANGE[1], C.F0_STEP))),
                                    k_max=C.K_MAX, snr_cap_db=C.SNR_CAP_DB, strong_db=C.STRONG_DB,
                                    mains_hz=C.MAINS_HZ, mains_mask_hw=C.MAINS_MASK_HW),
                      t2_physics_rule=dict(comb_snr_on_db=float(t2.t_on), f0_speed1_2_hz=float(t2.t12),
                                           f0_speed2_3_hz=float(t2.t23)))
    (C.MODEL_DIR / "final_model.json").write_text(json.dumps(model_json, indent=1))
    # per-trial reference for the MATLAB port: features + final-model probabilities (in-sample)
    # and the leave-one-round-out (out-of-sample) prediction of the same model type
    P_fin = final.predict_proba(X)
    ref = df[["trial", "cls"]].copy()
    for f in ("comb_snr", "f0", "n_harm", "line_snr"):
        ref[f] = df[f]
    for j, c in enumerate(C.CLASSES):
        ref[f"p_{c}"] = P_fin[:, j]
    ref["pred_final_in_sample"] = [C.CLASSES[k] for k in final.predict(X)]
    ref["pred_loro_out_of_sample"] = [C.CLASSES[k] for k in pred["HLR_aug"]]
    ref["p_true_loro"] = proba[np.arange(n), y]
    ref.to_csv(C.MODEL_DIR / "reference_predictions.csv", index=False)

    _fig_confusion(cms, res)
    _fig_window(win)
    _fig_final(final, X, y, bank)
    return dict(table=res.to_dict("records"), selected=sel.to_dict("records"), margins=margins,
                window=win.to_dict("records"), final_features=final.features,
                t2=model_json["t2_physics_rule"])


def _fig_confusion(cms, res):
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    for a, m in zip(ax.flat, ml.MODELS):
        cm = cms[m]
        a.imshow(cm, cmap="Blues", vmin=0, vmax=20)
        for i in range(4):
            for j in range(4):
                a.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 10 else "black")
        r = res[res.model == m].iloc[0]
        a.set(xticks=range(4), yticks=range(4), xticklabels=C.CLASSES, yticklabels=C.CLASSES,
              xlabel="predicted", ylabel="true",
              title=f"{ml.MODEL_LABEL[m]}\nLORO {r.loro_correct}/{r.loro_n}, 95 % CI {r.ci95_lo:.1%}-{r.ci95_hi:.1%}")
        a.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(C.FIG / "p3_confusion_clean.png", dpi=110)
    plt.close(fig)


def _fig_window(win):
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, d in win.groupby("model"):
        ax.errorbar(d.window_s, d.rep_mean * 100, yerr=d.rep_sd * 100, marker="o", capsize=3,
                    label=ml.MODEL_LABEL[m] + " (5-fold x10, mean +- sd)")
    ax.set(xscale="log", xlabel="analysed window [s]  (frequency resolution ~ 1/T)", ylabel="LORO accuracy [%]",
           title="Shorter windows: coarser spectra, f0 estimates blur")
    ax.set_xticks([0.5, 1, 2, 5])
    ax.set_xticklabels(["0.5", "1", "2", "5"])
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG / "p3_window_length.png", dpi=110)
    plt.close(fig)


def _fig_final(final, X, y, bank):
    cols = final.plot_cols
    bf = bank[0]
    Xa = bf.reshape(-1, bf.shape[2])
    ya = np.repeat(y, bf.shape[1])
    allx = np.vstack([X, Xa])
    lo, hi = np.percentile(allx[:, cols], [0.5, 99.5], axis=0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 300), np.linspace(lo[1], hi[1], 300))
    G = np.tile(X.mean(axis=0), (gx.size, 1))
    G[:, cols[0]], G[:, cols[1]] = gx.ravel(), gy.ravel()
    Z = final.predict(G).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(8, 6))
    from matplotlib.colors import ListedColormap
    ax.contourf(gx, gy, Z, levels=[-0.5, 0.5, 1.5, 2.5, 3.5], cmap=ListedColormap([C.CLASS_COLORS[c] for c in C.CLASSES]),
                alpha=0.15)
    rng = np.random.default_rng(0)
    jit = lambda v: v + (rng.uniform(-0.25, 0.25, v.shape) if np.allclose(v, np.round(v)) else 0)
    for c in range(4):
        ax.scatter(jit(Xa[ya == c, cols[0]]), Xa[ya == c, cols[1]], s=6, alpha=0.25, color=C.CLASS_COLORS[C.CLASSES[c]])
        ax.scatter(jit(X[y == c, cols[0]]), X[y == c, cols[1]], s=28, edgecolor="k", lw=0.5,
                   color=C.CLASS_COLORS[C.CLASSES[c]], label=C.CLASSES[c])
    others = [f for f in final.features if f not in (ml.NAMES[cols[0]], ml.NAMES[cols[1]])]
    extra = f" ({', '.join(others)} held at mean)" if others else ""
    ax.set(xlabel=ml.NAMES[cols[0]] + " (stage A: is the fan on?)", ylabel=ml.NAMES[cols[1]] + " (stage B: which speed?)",
           title=f"Final model decision regions{extra}\nlarge dots: real trials, small: noise-augmented copies")
    ax.title.set_fontsize(9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG / "p3_final_model_regions.png", dpi=110)
    plt.close(fig)
