"""Paper-sized versions of the robustness figures for the IEEE (two-column) LaTeX report.

The pipeline's p3_robustness_curves.png (5x5 grid) and p3_robustness_heatmap.png are drawn for the
screen; on an IEEE page they shrink to ~4 pt text. This script redraws them from the saved results
(results/tables/robustness*.csv) at the exact size of a landscape IEEE text block (696 x 516 pt minus
the caption), so they are placed at 100 % scale:
  p3_robustness_curves_part1.png / _part2.png   Fig. 16 split over two landscape pages (12 + 10 families)
  p3_robustness_heatmap_landscape.png           Fig. 17 on one landscape page
Same data, models, colours and shading as src/phase3_robustness.py.

    py report/fig_paper.py <output folder>
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as C  # noqa: E402
from src import noise_models as N  # noqa: E402
from src.calibrate_noise import load_cal  # noqa: E402
from src.phase3_robustness import EVAL_MODELS, STYLE, _lvl, in_band  # noqa: E402

OUT = pathlib.Path(sys.argv[1])
W, H = 9.6, 6.35          # inches: landscape text block (9.67 x 7.17 in) minus room for the caption
DPI = 220


def curves(res, cal, fams, path):
    nrow, ncol = 3, 4
    fig, axs = plt.subplots(nrow, ncol, figsize=(W, H))
    for k, (ax, f) in enumerate(zip(axs.flat, fams)):
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
            assert len(dm) == len(lv), (f, m)
            c, ls = STYLE[m]
            mu, sd = dm.acc.to_numpy() * 100, dm.sd.to_numpy() * 100
            ax.plot(xi, mu, ls, color=c, marker=".", ms=3, lw=1.5 if m == "HLR_aug" else 0.9, label=m)
            ax.fill_between(xi, mu - sd, mu + sd, color=c, alpha=0.10, lw=0)      # +-1 sd over repeats
        ax.set_xticks(xi)
        ax.set_xticklabels([_lvl(v) for v in lv], fontsize=6.5, rotation=30)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_ylim(0, 103)
        ax.axhline(25, color="k", lw=0.5, ls=":")
        tag = " (in-distribution)" if N.SPEC[f]["group"] == "mix" else ""
        ax.set_title(f + tag, fontsize=8.5, pad=3)
        ax.set_xlabel(N.SPEC[f]["unit"].replace("+-", "±").replace("m/s^2", "m/s²").replace("deg C", "°C"),
                      fontsize=7, labelpad=1)
        if k % ncol == 0:
            ax.set_ylabel("accuracy [%]", fontsize=7)
        ax.grid(alpha=0.3)
    for ax in axs.flat[len(fams):]:
        ax.axis("off")
    h, lab = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=len(lab), fontsize=7, frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=0.6, w_pad=0.8)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def heatmap(summ, fams, path):
    fig, ax = plt.subplots(1, 2, figsize=(W, H), layout="constrained")
    for a, col, sdcol, title in (
            (ax[0], "worst_realistic", "worst_realistic_sd", "worst accuracy within the calibrated realistic range"),
            (ax[1], "acc_harshest", "acc_harshest_sd", "accuracy at the harshest stress level tested")):
        M = summ.pivot(index="family", columns="model", values=col).loc[fams, EVAL_MODELS].to_numpy() * 100
        Sd = summ.pivot(index="family", columns="model", values=sdcol).loc[fams, EVAL_MODELS].to_numpy() * 100
        im = a.imshow(M, cmap="RdYlGn", vmin=25, vmax=100, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    a.text(j, i, f"{M[i, j]:.0f}±{Sd[i, j]:.0f}", ha="center", va="center", fontsize=5.6)
        a.set_xticks(range(len(EVAL_MODELS)))
        a.set_xticklabels(EVAL_MODELS, rotation=35, ha="right", fontsize=7)
        a.set_yticks(range(len(fams)))
        a.set_yticklabels(fams, fontsize=7)
        a.set_title(title + " (mean ± sd)", fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("accuracy [%]", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(C.TAB / "robustness.csv")
    summ = pd.read_csv(C.TAB / "robustness_summary.csv")
    cal = load_cal()
    fams = list(N.SPEC)
    curves(res, cal, fams[:12], OUT / "p3_robustness_curves_part1.png")
    curves(res, cal, fams[12:], OUT / "p3_robustness_curves_part2.png")
    heatmap(summ, fams, OUT / "p3_robustness_heatmap_landscape.png")
    print("wrote", OUT)
