"""Single entry point: data/raw -> results/.

    py run_all.py              full pipeline (~8 min on 16 cores)
    py run_all.py --quick      robustness study with 2 instead of 5 CV repeats
    py run_all.py --no-robust  skip the noise-robustness study

Phases
  1   acquisition audit + preprocessing justification          (Week 1)
  2   features, FFT / CWT / time plots, discriminability       (Weeks 2-3)
  cal noise-model calibration against the real recordings
  3a  classification + validation on the real recordings       (Weeks 3-4)
  3b  robustness to calibrated noise, ablation
  3c  feature-selection design study (why invariant features + worst-group selection)
"""
import argparse
import json
import shutil
import time

import numpy as np

from src import calibrate_noise, config as C, load_trials as L, phase1_audit, phase2_features
from src import phase3_classify, phase3_robustness, phase3_selection_study, report


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-robust", action="store_true")
    args = ap.parse_args()
    for d in (C.RESULTS, C.FIG, C.TAB, C.MODEL_DIR, C.RESULTS / "summaries"):
        d.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(C.RESULTS / "cache", ignore_errors=True)        # never reuse stale augmentation

    def save(name, d):
        (C.RESULTS / "summaries" / f"{name}.json").write_text(json.dumps(d, indent=1, default=_default))

    t0 = time.time()
    trials = L.load_all()
    print(f"{len(trials)} trials loaded from {C.DATA_RAW}")
    print("[1/5] Phase 1: audit + preprocessing ...")
    save("phase1", phase1_audit.run(trials))
    print("[2/5] Phase 2: features + plots ...")
    save("phase2", phase2_features.run(trials))
    print("[3/5] Noise calibration ...")
    calibrate_noise.calibrate(trials)
    print("[4/5] Phase 3a: classification on real recordings ...")
    s3 = phase3_classify.run()
    save("phase3_clean", s3)
    if not args.no_robust:
        print("[5/5] Phase 3b: noise robustness (this is the long one) ...")
        phase3_robustness.run(R=2 if args.quick else 5)
        print("      Phase 3c: feature-selection design study ...")
        phase3_selection_study.run(R=2)
    path = report.write()
    print(f"done in {time.time() - t0:.0f} s -> {path}")


if __name__ == "__main__":
    main()
