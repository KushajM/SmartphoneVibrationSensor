"""Phase 3 machine-learning core.

Models (all use 2-3 features, as the brief requires)
  T1_textbook   threshold rule: z-axis RMS (off/on) + FFT dominant frequency (speed)
  T2_physics    threshold rule: comb SNR with a constant-false-alarm threshold (off/on) +
                comb fundamental f0 (speed)
  kNN_clean     k-nearest-neighbours, features chosen by nested selection on clean data
  LR_clean      multinomial logistic regression, same
  kNN_aug       kNN trained on clean + noise-augmented copies of the training trials,
                features chosen by nested selection on that augmented data
  LR_aug        logistic regression, same
  HLR_clean     two-stage ("detect, then estimate") logistic regression, clean training
  HLR_aug       two-stage logistic regression, noise-augmented training   <- FINAL MODEL
                stage A: binary LR, fan off vs on   (<= 2 selected features)
                stage B: 3-class LR among speeds    (<= 1 selected feature, trained on on-trials)
                A single multinomial LR cannot express "f0 only matters when the fan is on"
                (an off trial's f0 is random), which is exactly the structure of the problem.

Leakage rules enforced here
  * every split is by TRIAL; augmented copies of a trial are only ever used while that trial
    is in the training part; inner selection folds are also split by trial
  * thresholds, scalers, feature choice and model weights are fitted on training data only
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features as F

NAMES = F.FEATURE_NAMES
IDX = {n: i for i, n in enumerate(NAMES)}
# Features invariant BY CONSTRUCTION to (a) the vibration gain (mount tightness, wall, phone
# mass, sensor sensitivity) - ratios to the local noise floor, frequencies, shape statistics -
# and (b) phone orientation (vector / principal-axis quantities). Raw amplitudes (RMS, p2p,
# FFT amplitude) and single-axis features are excluded from the robust models' candidate pool.
INVARIANT = ["comb_snr", "f0", "n_harm", "line_snr", "spec_flatness", "spec_centroid", "band_ratio",
             "welch_fdom", "fft_fdom_vec", "kurt_pc1", "crest_pc1"]

MODELS = ["T1_textbook", "T2_physics", "kNN_clean", "kNN_aug", "LR_clean", "LR_aug", "HLR_clean", "HLR_aug"]
MODEL_LABEL = {
    "T1_textbook": "T1 textbook threshold (z-RMS + FFT peak)",
    "T2_physics": "T2 physics threshold (comb SNR + f0)",
    "kNN_clean": "kNN, trained on clean data",
    "LR_clean": "LogReg, trained on clean data",
    "kNN_aug": "kNN, noise-augmented training",
    "LR_aug": "LogReg, noise-augmented training",
    "HLR_clean": "2-stage LogReg, trained on clean data",
    "HLR_aug": "2-stage LogReg, noise-augmented training (FINAL)",
}


# ------------------------------------------------------------------ statistics
def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def confusion(y, p, n=4):
    cm = np.zeros((n, n), int)
    np.add.at(cm, (y, p), 1)
    return cm


# ------------------------------------------------------------------ threshold rules
def margin_thr(lo, hi):
    """Max-margin cut between two 1-D classes (midpoint of the gap); median midpoint if they
    overlap."""
    return (lo.max() + hi.min()) / 2 if lo.max() < hi.min() else (np.median(lo) + np.median(hi)) / 2


def cfar_thr(off, on):
    """Constant-false-alarm-rate cut set from the OFF class only (mean + 4 sd, at least
    max + 0.5 dB). The comb SNR of an off trial is noise-versus-noise, so its distribution does
    not move when noise rises; a midpoint would sit needlessly close to the on class."""
    t = max(off.mean() + 4 * off.std(), off.max() + 0.5)
    return t if t < on.min() else margin_thr(off, on)


class ThresholdRule:
    def __init__(self, on_feat, speed_feat, rule="margin"):
        self.on, self.sp, self.rule = IDX[on_feat], IDX[speed_feat], rule
        self.features = [on_feat, speed_feat]

    def fit(self, X, y):
        a, s = X[:, self.on], X[:, self.sp]
        self.t_on = (cfar_thr if self.rule == "cfar" else margin_thr)(a[y == 0], a[y > 0])
        self.t12 = margin_thr(s[y == 1], s[y == 2])
        self.t23 = margin_thr(s[y == 2], s[y == 3])
        return self

    def predict(self, X):
        s = X[:, self.sp]
        p = np.where(s < self.t12, 1, np.where(s < self.t23, 2, 3))
        return np.where(X[:, self.on] < self.t_on, 0, p)

    def margins(self, X):
        """Signed distance to the nearest decision boundary (feature units)."""
        a, s = X[:, self.on], X[:, self.sp]
        return a - self.t_on, np.minimum(np.abs(s - self.t12), np.abs(s - self.t23))


# ------------------------------------------------------------------ selected-feature ML
def make_estimator(kind):
    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))


def fisher_ratio(X, y):
    mu = X.mean(axis=0)
    num = np.zeros(X.shape[1])
    den = np.zeros(X.shape[1])
    for c in np.unique(y):
        Xc = X[y == c]
        num += len(Xc) * (Xc.mean(axis=0) - mu) ** 2
        den += ((Xc - Xc.mean(axis=0)) ** 2).sum(axis=0)
    return num / np.maximum(den, 1e-30)


def _trial_folds(groups, y, k, seed):
    rng = np.random.default_rng(seed)
    g = np.unique(groups)
    lab = np.array([y[groups == gi][0] for gi in g])
    fold = np.empty(len(g), int)
    for c in np.unique(lab):
        idx = rng.permutation(np.where(lab == c)[0])
        fold[idx] = np.arange(idx.size) % k
    return {gi: f for gi, f in zip(g, fold)}


def select_features(X, y, groups, kind, k_max=3, n_pre=12, n_inner=3, seed=0, masks=None, pool=None):
    """Greedy forward selection (<= k_max features), inner folds split by TRIAL.

    Objective
      masks is None : inner-CV accuracy (plain).
      masks given   : WORST-GROUP inner-CV accuracy (distributionally robust selection). Each
                      training row carries a bitmask of the noise groups used to corrupt it
                      (0 = real trial). Accuracy is computed separately on the clean rows and on
                      the rows touched by each noise group, and the MINIMUM is maximised (ties ->
                      mean accuracy -> Fisher ratio). A feature that only works under some noise
                      types is rejected, which is what lets the model generalise to noise types
                      it never saw (checked with leave-one-group-out tests).
    Candidates are pre-filtered to the n_pre highest Fisher ratios on the training rows."""
    fr = fisher_ratio(X, y)
    allowed = set(range(X.shape[1])) if pool is None else {IDX[f] for f in pool}
    cand = [int(j) for j in np.argsort(fr)[::-1] if int(j) in allowed][:n_pre]
    fmap = _trial_folds(groups, y, n_inner, seed)
    fold = np.array([fmap[g] for g in groups])
    cells = None
    if masks is not None:
        cells = [masks == 0] + [((masks >> b) & 1).astype(bool) for b in range(int(masks.max()).bit_length())]
        cells = [c for c in cells if c.sum() >= 10]

    def score(cols):
        pred = np.empty_like(y)
        for f in range(n_inner):
            tr, te = fold != f, fold == f
            pred[te] = make_estimator(kind).fit(X[tr][:, cols], y[tr]).predict(X[te][:, cols])
        hit = pred == y
        if cells is None:
            return (hit.mean(), hit.mean())
        return (min(hit[c].mean() for c in cells), hit.mean())

    chosen, best = [], (-1.0, -1.0)
    while len(chosen) < k_max:
        sc = {j: score(chosen + [j]) for j in cand if j not in chosen}
        j = max(sc, key=lambda q: (sc[q][0], sc[q][1], fr[q]))
        if chosen and sc[j][0] <= best[0] + 1e-3:
            break
        chosen.append(j)
        best = sc[j]
    return chosen, best[1]


class SelectedModel:
    def __init__(self, kind, seed=0, pool=None, worst_group=False):
        self.kind, self.seed, self.pool, self.worst_group = kind, seed, pool, worst_group

    def fit(self, X, y, groups, masks=None):
        self.cols, self.inner_acc = select_features(X, y, groups, self.kind, seed=self.seed,
                                                    masks=masks if self.worst_group else None, pool=self.pool)
        self.plot_cols = self.cols[:2]
        self.features = [NAMES[j] for j in self.cols]
        self.est = make_estimator(self.kind).fit(X[:, self.cols], y)
        return self

    def predict(self, X):
        return self.est.predict(X[:, self.cols])

    def predict_proba(self, X):
        return self.est.predict_proba(X[:, self.cols])


class TwoStageLR:
    """Detect, then estimate. P(off) = 1 - pA ; P(speed k) = pA * pB_k."""

    def __init__(self, seed=0, kmax_a=2, kmax_b=1, pool=None, worst_group=False):
        self.seed, self.kmax_a, self.kmax_b = seed, kmax_a, kmax_b
        self.pool, self.worst_group = pool, worst_group

    def fit(self, X, y, groups, masks=None):
        masks = masks if self.worst_group else None
        on = (y > 0).astype(int)
        self.ca, _ = select_features(X, on, groups, "lr", k_max=self.kmax_a, seed=self.seed, masks=masks,
                                     pool=self.pool)
        self.A = make_estimator("lr").fit(X[:, self.ca], on)
        m = y > 0
        self.cb, _ = select_features(X[m], y[m], groups[m], "lr", k_max=self.kmax_b, seed=self.seed,
                                     masks=None if masks is None else masks[m], pool=self.pool)
        self.B = make_estimator("lr").fit(X[m][:, self.cb], y[m])
        self.cols = list(dict.fromkeys(self.ca + self.cb))
        self.plot_cols = [self.ca[0], self.cb[0]]
        self.features = [NAMES[j] for j in self.cols]
        self.features_a = [NAMES[j] for j in self.ca]
        self.features_b = [NAMES[j] for j in self.cb]
        return self

    def predict_proba(self, X):
        pa = self.A.predict_proba(X[:, self.ca])[:, 1]
        pb = self.B.predict_proba(X[:, self.cb])
        return np.column_stack([1 - pa, pa[:, None] * pb])

    def predict(self, X):
        on = self.A.predict(X[:, self.ca]) == 1
        return np.where(on, self.B.predict(X[:, self.cb]), 0)


def training_rows(clean, bank, train_idx, y):
    """Stack clean rows (+ augmented copies) of the training trials. Returns X, y, trial id and
    noise-group bitmask (0 for real rows). bank = (features n x K x F, masks n x K) or None."""
    X, Y, G, M = [clean[train_idx]], [y[train_idx]], [train_idx], [np.zeros(len(train_idx), int)]
    if bank is not None:
        bf, bm = bank
        K = bf.shape[1]
        X.append(bf[train_idx].reshape(-1, bf.shape[2]))
        Y.append(np.repeat(y[train_idx], K))
        G.append(np.repeat(train_idx, K))
        M.append(bm[train_idx].reshape(-1))
    return np.vstack(X), np.concatenate(Y), np.concatenate(G), np.concatenate(M)


# Selection procedure of the *_aug models. Chosen by a leave-one-noise-group-out comparison
# (mean accuracy over 15 unseen-noise tests: kNN 0.78 -> 0.87, 2-stage LR 0.84 -> 0.85):
ROBUST_POOL = INVARIANT      # invariant-by-construction candidate features only
WORST_GROUP = True           # maximise the worst noise-group accuracy, not the mean


def fit_model(name, clean, y, train_idx, bank=None, seed=0, pool="default", worst_group=None):
    if name == "T1_textbook":
        return ThresholdRule("rms_z", "fft_fdom_z", "margin").fit(clean[train_idx], y[train_idx])
    if name == "T2_physics":
        return ThresholdRule("comb_snr", "f0", "cfar").fit(clean[train_idx], y[train_idx])
    aug = name.endswith("_aug")
    X, Y, G, M = training_rows(clean, bank if aug else None, train_idx, y)
    pool = (ROBUST_POOL if aug else None) if pool == "default" else pool
    wg = (WORST_GROUP and aug) if worst_group is None else worst_group
    if name.startswith("HLR"):
        return TwoStageLR(seed, pool=pool, worst_group=wg).fit(X, Y, G, M)
    return SelectedModel("knn" if name.startswith("kNN") else "lr", seed, pool=pool, worst_group=wg).fit(X, Y, G, M)


# ------------------------------------------------------------------ CV schemes
def loro_folds(rounds):
    """Leave-one-round-out: the 4 trials recorded consecutively in one round-robin cycle
    (one per class) are held out together -> no temporal neighbour of a test trial is trained on."""
    return [(np.where(rounds != r)[0], np.where(rounds == r)[0]) for r in np.unique(rounds)]


def stratified_folds(y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    f = np.empty(len(y), int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        f[idx] = np.arange(idx.size) % k
    return [(np.where(f != i)[0], np.where(f == i)[0]) for i in range(k)]
