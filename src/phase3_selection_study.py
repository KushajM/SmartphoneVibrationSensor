"""Phase 3c - why the robust models use invariant features + worst-group selection.

Compares four feature-selection procedures for the noise-augmented models under the strict
leave-one-noise-group-out protocol (the tested noise type never appears in training):
    all features      / maximise mean inner-CV accuracy
    all features      / maximise worst-noise-group inner-CV accuracy
    invariant features/ maximise mean inner-CV accuracy
    invariant features/ maximise worst-noise-group inner-CV accuracy   <- adopted
Output: tables/selection_study.csv
"""
import numpy as np
import pandas as pd
from joblib import delayed

from . import config as C
from . import ml
from . import noise_models as N
from . import phase3_classify as P3
from . import workers as W

TESTS = [("clean", 0), ("gaussian", -10), ("gaussian", -15), ("jn_proofmass", 30), ("lower_fs", 200),
         ("quantize", 0.019), ("appliance", 0.3), ("coupling", 0.15), ("knocks", 1.0), ("footsteps", 0.3),
         ("salt_pepper", 0.01), ("speckle", 1.0), ("dropouts", 30), ("popcorn", 0.1), ("mix_same_day", 2),
         ("mix_other_phone", 1)]
CONFIGS = {"all features, mean accuracy": (None, False), "all features, worst-group": (None, True),
           "invariant, mean accuracy": ("inv", False), "invariant, worst-group (adopted)": ("inv", True)}


def _eval(cfg, model, f, lv, X, y, bank, feats, R):
    pool, wg = CONFIGS[cfg]
    pool = ml.INVARIANT if pool == "inv" else None
    acc = []
    for r in range(R):
        ok = 0
        for k, (tr, te) in enumerate(ml.stratified_folds(y, 5, seed=100 + r)):
            m = ml.fit_model(model, X, y, tr, bank, seed=100 * r + k, pool=pool, worst_group=wg)
            Xs = X if f == "clean" else feats[(f, lv, r)]
            ok += int((m.predict(Xs[te]) == y[te]).sum())
        acc.append(ok / len(y))
    return dict(config=cfg, model=model, test=f"{f} @ {lv:g}", acc=float(np.mean(acc)), sd=float(np.std(acc)))


def run(R=2):
    df, X, y, _ = P3.load_table()
    n = len(y)
    groups = ["all"] + sorted({s["group"] for s in N.SPEC.values()} - {"mix"})
    banks = {g: P3.bank_cache(g, n) for g in groups}
    keys = [(f, lv, r) for f, lv in TESTS if f != "clean" for r in range(R)]
    feats = dict(zip(keys, W.run([delayed(W.corrupt_task)(f, lv, r) for f, lv, r in keys])))
    jobs = []
    for cfg in CONFIGS:
        for model in ("HLR_aug", "kNN_aug"):
            for f, lv in TESTS:
                g = "all" if f == "clean" or N.SPEC[f]["group"] == "mix" else N.SPEC[f]["group"]
                jobs.append(delayed(_eval)(cfg, model, f, lv, X, y, banks[g],
                                           {k: v for k, v in feats.items() if k[0] == f and k[1] == lv}, R))
    res = pd.DataFrame(W.run(jobs))
    res.to_csv(C.TAB / "selection_study.csv", index=False)
    return res
