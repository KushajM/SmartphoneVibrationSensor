"""Collect every number the Word report quotes into report/report_data.json (so the report is
always generated from the current results, never retyped).  Run after run_all.py:
    py report/build_report_data.py
"""
import json
import platform
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config as C  # noqa: E402
from src import ml  # noqa: E402
from src import noise_models as N  # noqa: E402

R = C.RESULTS
T = C.TAB


def jl(p):
    return json.loads(Path(p).read_text())


def num(x):
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def records(df):
    return [{k: num(v) for k, v in r.items()} for r in df.to_dict("records")]


D = {}
D["p1"] = jl(R / "summaries" / "phase1.json")
D["p2"] = jl(R / "summaries" / "phase2.json")
D["p3"] = jl(R / "summaries" / "phase3_clean.json")
cal = jl(C.CAL_FILE)
D["cal_evidence"] = cal["evidence"]
D["cal_bands"] = cal["bands"]
D["cal_table"] = records(pd.read_csv(T / "noise_calibration.csv"))
D["freq"] = records(pd.read_csv(T / "frequency_summary.csv"))

rank = pd.read_csv(T / "feature_ranking.csv")
D["ranking"] = records(rank[["feature", "fisher", "worst_pair_gap_sd", "separable_pairs"]])
key = ["f0", "comb_snr", "n_harm", "line_snr", "fft_fdom_vec", "cwt_fdom", "rms_z", "rms_vec", "kurt_pc1",
       "crest_pc1", "spec_flatness", "band_ratio"]
pc = []
for f in key:
    r = rank[rank.feature == f].iloc[0]
    pc.append(dict(feature=f, **{c: [num(r[f"mean_{c}"]), num(r[f"sd_{c}"])] for c in C.CLASSES}))
D["per_class"] = pc

D["clean_cv"] = records(pd.read_csv(T / "clean_cv.csv"))
D["selected"] = records(pd.read_csv(T / "selected_features.csv"))
D["window"] = records(pd.read_csv(T / "window_length.csv"))
D["edge"] = records(pd.read_csv(T / "edge_check.csv").describe().loc[["mean", "std", "min", "max"]].reset_index())

rs = pd.read_csv(T / "robustness_summary.csv")
D["rob_summary"] = records(rs)
rob = pd.read_csv(T / "robustness.csv")
D["rob_full"] = records(rob[rob.model.isin(["T1_textbook", "T2_physics", "HLR_aug", "HLR_aug_known"])])
D["ablation"] = records(pd.read_csv(T / "ablation.csv"))
D["selection"] = records(pd.read_csv(T / "selection_study.csv"))
D["families"] = {k: dict(unit=v["unit"], group=v["group"], levels=[None if not np.isfinite(x) else float(x)
                                                                     for x in v["levels"]],
                         aug=None if v["aug"] is None else [str(v["aug"][0])] + [x if isinstance(x, list) else float(x) for x in v["aug"][1:]])
                 for k, v in N.SPEC.items()}
D["model"] = jl(C.MODEL_DIR / "final_model.json")

ref = pd.read_csv(C.MODEL_DIR / "reference_predictions.csv")
demo = pd.read_csv(T / "demo_predictions.csv")
conf = demo[["p_off", "p_speed1", "p_speed2", "p_speed3"]].max(axis=1)
D["demo"] = dict(n=len(demo), correct=int((demo.true == demo.predicted).sum()),
                 conf_mean=float(conf.mean()), conf_sd=float(conf.std(ddof=0)), conf_min=float(conf.min()),
                 loro_correct=int((ref.cls == ref.pred_loro_out_of_sample).sum()),
                 loro_ptrue_mean=float(ref.p_true_loro.mean()), loro_ptrue_sd=float(ref.p_true_loro.std(ddof=0)),
                 loro_ptrue_min=float(ref.p_true_loro.min()),
                 per_class=[dict(cls=c, conf_mean=float(conf[demo.true == c].mean()),
                                 conf_sd=float(conf[demo.true == c].std(ddof=0)),
                                 f0_mean=float(demo.f0[demo.true == c].mean()),
                                 f0_sd=float(demo.f0[demo.true == c].std(ddof=0)),
                                 nh_mean=float(demo.n_harm[demo.true == c].mean())) for c in C.CLASSES])
log = (R / "demo_predictions_log.txt").read_text().splitlines()
D["demo_log"] = log[:16] + ["..."] + log[-4:]

# noise demo: 9 scenarios x (speed2_05, off_05)
scen = ["footsteps 0.3", "knocks 1.0", "gaussian -10", "salt_pepper 0.01", "mains_hum 4", "orientation 90",
        "lower_fs 200", "appliance 1", "speed_shift 4"]
arrows = [l for l in (R / "demo_noise_log.txt").read_text().splitlines() if "->" in l]
nd = []
for i, s in enumerate(scen):
    a, b = arrows[2 * i], arrows[2 * i + 1]
    parse = lambda l: (l.split("->")[1].split("|")[0].strip().lower(), l.split("|")[1].strip())
    nd.append(dict(noise=s, speed2=parse(a), off=parse(b)))
D["noise_demo"] = nd

ml_log = (R / "matlab_demo_log.txt").read_text()
D["matlab"] = dict(
    seconds=float(re.search(r"(\d+) recordings in ([\d.]+) s", ml_log).group(2)),
    df0=re.search(r"f0 difference\|\s+=\s+(\S+)", ml_log).group(1),
    dcs=re.search(r"comb SNR difference\|\s+=\s+(\S+)", ml_log).group(1),
    nh=re.search(r"mismatches\s+=\s+(\d+)", ml_log).group(1),
    dp=re.search(r"probability diff\|\s+=\s+(\S+)", ml_log).group(1),
    same=re.search(r"identical predictions\s+=\s+(\S+)", ml_log).group(1),
    result=re.search(r"RESULT: (\w+)", ml_log).group(1),
    excerpt=[l for l in ml_log.splitlines() if l.strip()][2:14])

import scipy  # noqa: E402
import sklearn  # noqa: E402
import matplotlib  # noqa: E402
D["env"] = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, matplotlib=matplotlib.__version__, pandas=pd.__version__,
                matlab="R2018a (9.4.0.813654)", os=platform.platform())
D["config"] = {k: getattr(C, k) for k in ("FS", "TRIM_S", "SINC_HALF", "SINC_BETA", "HP_ORDER", "HP_FC",
                                          "HAMPEL_HALF", "HAMPEL_NSIG", "WELCH_NPERSEG", "ROB_NPERSEG",
                                          "FLOOR_HALFWIDTH_HZ", "F0_RANGE", "K_MAX", "SNR_CAP_DB", "STRONG_DB",
                                          "CWT_CYCLES")}
D["models"] = ml.MODELS
out = ROOT / "report" / "report_data.json"
out.write_text(json.dumps(D, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
print("wrote", out)
