"""Classify Phyphox recordings with the final model (for the in-class demo).

usage:
  py -m src.predict  rec.zip [rec2.zip ...]          classify recordings (.zip export or Raw Data .csv)
  py -m src.predict  *.zip --csv out.csv            batch + save a table (+ accuracy if names are <class>_NN)
  py -m src.predict  speed2_05.zip --noise footsteps 0.3     corrupt with a calibrated noise model first
                                                    (families: see src/noise_models.py SPEC)

The model is read from results/model/final_model.json (written by run_all.py). Prediction needs
only numpy/scipy; the same JSON drives the MATLAB port (matlab/vib_predict.m).
"""
import argparse
import glob
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from . import config as C
from . import features as F
from . import preprocess as P


def load_any(path):
    if str(path).lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            d = np.loadtxt(io.TextIOWrapper(z.open("Raw Data.csv"), encoding="utf-8"), delimiter=",", skiprows=1)
    else:
        d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1:4]


def _stage(st, feats):
    z = (np.array([feats[f] for f in st["features"]]) - np.array(st["scaler_mean"])) / np.array(st["scaler_scale"])
    W, b = np.array(st["coef"]), np.array(st["intercept"])
    if W.shape[0] == 1:                                   # binary logistic regression
        p1 = 1.0 / (1.0 + np.exp(-(W @ z + b)[0]))
        return np.array([1 - p1, p1])
    s = W @ z + b                                         # multinomial (softmax)
    e = np.exp(s - s.max())
    return e / e.sum()


def predict_arrays(t, a, model):
    dur = t[-1] - t[0]
    trim = C.TRIM_S if dur >= C.TRIM_S + 1.0 else max(1.0, dur - 1.0)
    feats = F.extract(P.preprocess(t, a, trim=trim))
    pa = _stage(model["stage_a"], feats)[1]              # P(fan on)
    pb = _stage(model["stage_b"], feats)                 # P(speed | on)
    speeds = [C.CLASSES[c] for c in model["stage_b"]["classes"]]
    probs = {"off": 1 - pa, **{s: pa * p for s, p in zip(speeds, pb)}}
    label = "off" if pa < 0.5 else speeds[int(np.argmax(pb))]
    return label, probs, feats, trim


def predict(path, model=None, noise=None, seed=0):
    model = model or json.loads((C.MODEL_DIR / "final_model.json").read_text())
    t, a = load_any(path)
    if noise:
        from . import noise_models as N
        from .calibrate_noise import load_cal
        t, a = N.apply(noise[0], np.random.default_rng(seed), t, a, float(noise[1]), load_cal())
    return predict_arrays(t, a, model)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--noise", nargs=2, metavar=("FAMILY", "LEVEL"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv")
    args = ap.parse_args(argv)
    paths = [p for f in args.files for p in (sorted(glob.glob(f)) or [f])]
    model = json.loads((C.MODEL_DIR / "final_model.json").read_text())
    print(f"model: {model['model']}")
    print(f"       stage A features {model['stage_a']['features']}, stage B features {model['stage_b']['features']}")
    if args.noise:
        print(f"       input corrupted with calibrated noise: {args.noise[0]} at level {args.noise[1]} (seed {args.seed})")
    print()
    rows = []
    for p in paths:
        label, probs, f, trim = predict(p, model, args.noise, args.seed)
        pr = "  ".join(f"{k} {v:5.1%}" for k, v in probs.items())
        print(f"{p}\n  -> {label.upper():7s} | {pr}")
        if label == "off":
            msg = f"no fan harmonic >= 6 dB above the noise floor (comb SNR {f['comb_snr']:.1f} dB)"
        else:
            msg = (f"f0 = {f['f0']:.2f} Hz ({60 * f['f0']:.0f} rpm if f0 = shaft rate), dominant line "
                   f"{3 * f['f0']:.1f} Hz, harmonics >= 6 dB: {int(f['n_harm'])}, comb SNR {f['comb_snr']:.1f} dB")
        print("     " + msg + ("" if trim == C.TRIM_S else f"  [short recording: {trim:.1f} s window]"))
        stem = Path(p).stem
        true = stem.rsplit("_", 1)[0] if stem.rsplit("_", 1)[0] in C.CLASSES else ""
        rows.append(dict(file=Path(p).name, true=true, predicted=label, **{f"p_{k}": v for k, v in probs.items()},
                         f0=f["f0"], n_harm=int(f["n_harm"]), comb_snr=f["comb_snr"]))
    known = [r for r in rows if r["true"]]
    if len(rows) > 1 and known:
        ok = sum(r["true"] == r["predicted"] for r in known)
        conf = np.array([max(r[f"p_{r['predicted']}"] for _ in [0]) for r in rows])
        print(f"\n{ok}/{len(known)} correct; confidence of predicted class: mean {conf.mean():.1%} +- {conf.std():.1%}, "
              f"min {conf.min():.1%}")
    if args.csv:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"table written to {args.csv}")


if __name__ == "__main__":
    main()
