"""Load Phyphox trials (data/raw/<class>/<class>_NN.csv) and their metadata (meta/ inside the
original zip exports), and compute the per-trial acquisition audit used for the Week-1 note."""
import csv
import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from . import config as C


@dataclass
class Trial:
    name: str
    cls: str
    label: int
    nn: int                       # trial number = recording round (round-robin order)
    t: np.ndarray                 # timestamps [s]
    a: np.ndarray                 # N x 3 acceleration incl. gravity [m/s^2]
    meta: dict = field(default_factory=dict)


def read_csv(path):
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    return d[:, 0], d[:, 1:4]


def _find_zip(name):
    for d in C.ZIP_DIRS:
        p = d / f"{name}.zip"
        if p.exists():
            return p
    return None


def read_meta(zpath):
    """START time and device/sensor description from a Phyphox export."""
    meta = {}
    if zpath is None:
        return meta
    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        if "meta/time.csv" in names:
            for row in csv.reader(io.TextIOWrapper(z.open("meta/time.csv"), encoding="utf-8")):
                if row and row[0] == "START":
                    meta["start_epoch"] = float(row[2])
                    meta["start_text"] = row[3]
        if "meta/device.csv" in names:
            dev = {r[0]: r[1] for r in csv.reader(io.TextIOWrapper(z.open("meta/device.csv"), encoding="utf-8"))
                   if len(r) >= 2}
            meta["device"] = dev.get("deviceModel", "")
            meta["sensor"] = f'{dev.get("accelerometer Vendor", "")} {dev.get("accelerometer Name", "")}'.strip()
            for k, key in [("range", "accelerometer Range"), ("resolution", "accelerometer Resolution"),
                           ("min_delay_us", "accelerometer MinDelay")]:
                try:
                    meta[k] = float(dev.get(key, "nan"))
                except ValueError:
                    meta[k] = float("nan")
    return meta


def load_all(classes=C.CLASSES):
    trials = []
    for lab, c in enumerate(classes):
        for p in sorted((C.DATA_RAW / c).glob(f"{c}_*.csv")):
            t, a = read_csv(p)
            name = p.stem
            trials.append(Trial(name=name, cls=c, label=lab, nn=int(name.split("_")[-1]),
                                t=t, a=a, meta=read_meta(_find_zip(name))))
    if not trials:
        raise FileNotFoundError(f"no trials found under {C.DATA_RAW}")
    return trials


def audit(tr: Trial):
    """Acquisition quality metrics for one trial.

    Timing: Android delivers each sample with a timestamp; the sensor itself is clocked by a
    hardware oscillator (output data rate, ODR). We assign every sample to an integer clock slot
    and regress time on slot index: the slope gives the true rate (fs_fit), the residual is pure
    timestamp noise, and missing slots are dropped samples.
    """
    t, a = tr.t, tr.a
    dt = np.diff(t)
    dt_med = np.median(dt)
    slots = np.round((t - t[0]) / dt_med).astype(np.int64)
    slope, icpt = np.polyfit(slots, t, 1)
    resid = t - (slope * slots + icpt)
    n_slots = int(slots[-1]) + 1
    g = a.mean(axis=0)
    gn = np.linalg.norm(g)
    spacing = []
    for k in range(3):
        u = np.unique(np.round(a[:, k], 7))
        du = np.diff(u)
        du = du[du > 1e-6]
        spacing.append(np.median(np.sort(du)[:20]) if du.size else np.nan)
    row = dict(
        trial=tr.name, cls=tr.cls, round=tr.nn,
        start=tr.meta.get("start_text", ""), start_epoch=tr.meta.get("start_epoch", np.nan),
        n_samples=len(t), duration_s=t[-1] - t[0],
        fs_median=1.0 / dt_med, fs_fit=1.0 / slope,
        ts_resid_us=resid.std() * 1e6, dt_std_us=dt.std() * 1e6, max_dt_ms=dt.max() * 1e3,
        dropped_samples=n_slots - len(t),
        g_x=g[0], g_y=g[1], g_z=g[2], g_norm=gn,
        tilt_inplane_deg=np.degrees(np.arctan2(g[0], g[1])),     # rotation about the wall normal
        tilt_outplane_deg=np.degrees(np.arcsin(g[2] / gn)),       # lean away from the wall
        value_spacing=float(np.nanmin(spacing)),
    )
    row["flag"] = ";".join(f for f, bad in [
        ("short", row["duration_s"] < 6.5),
        ("gaps", row["dropped_samples"] > 0),
        ("fs_outlier", abs(row["fs_fit"] - 400.1) > 1.0),
    ] if bad)
    return row


def start_datetime(tr: Trial):
    e = tr.meta.get("start_epoch")
    return datetime.fromtimestamp(e) if e else None
