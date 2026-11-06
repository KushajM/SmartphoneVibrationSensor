"""Parallel feature computation (joblib/loky). Each worker process loads the trials and the
calibration once; tasks only carry small arguments. All randomness is seeded from a hash of
(task, trial, copy) so every run reproduces bit-for-bit."""
import zlib

import numpy as np
from joblib import Parallel, delayed

from . import config as C
from . import features as F
from . import load_trials as L
from . import noise_models as N
from . import preprocess as P

_CTX = {}


def _ctx():
    if not _CTX:
        from .calibrate_noise import load_cal
        _CTX["trials"] = L.load_all()
        _CTX["cal"] = load_cal() if C.CAL_FILE.exists() else None
    return _CTX["trials"], _CTX["cal"]


def _seed(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode())


def _feat(t, a, despike=True, trim=C.TRIM_S, opts=None):
    return F.vector(F.extract(P.preprocess(t, a, despike=despike, trim=trim), opts=opts))


def clean_task(i, despike=True, trim=C.TRIM_S, opts=None):
    trials, _ = _ctx()
    return _feat(trials[i].t, trials[i].a, despike, trim, opts)


def corrupt_task(family, level, r, despike=True, opts=None):
    """Features of all trials, each corrupted by `family` at `level` (repeat r)."""
    trials, cal = _ctx()
    out = []
    for i, tr in enumerate(trials):
        rng = np.random.default_rng(_seed(family, level, r, i))
        t, a = N.apply(family, rng, tr.t, tr.a, level, cal)
        out.append(_feat(t, a, despike, opts=opts))
    return np.array(out)


def augment_task(i, exclude_group, K):
    """K noise-augmented feature vectors of trial i, never using `exclude_group`."""
    trials, cal = _ctx()
    ex = () if exclude_group is None else (exclude_group,)
    out, masks = [], []
    for j in range(K):
        rng = np.random.default_rng(_seed("aug", exclude_group, i, j))
        t, a, mk = N.augment(rng, trials[i].t, trials[i].a, cal, ex, return_mask=True)
        out.append(_feat(t, a))
        masks.append(mk)
    return np.array(out), np.array(masks)


def run(tasks, n_jobs=-2, verbose=0):
    return Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(tasks)


def clean_features(n, **kw):
    return np.array(run([delayed(clean_task)(i, **kw) for i in range(n)]))


def augmentation_bank(n, exclude_group, K=4):
    """Returns (features n x K x F, noise-group bitmasks n x K)."""
    res = run([delayed(augment_task)(i, exclude_group, K) for i in range(n)])
    return np.array([r[0] for r in res]), np.array([r[1] for r in res])
