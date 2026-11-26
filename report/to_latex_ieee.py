"""Convert the dumped Final_Report.docx (user.txt) into a single main.tex for the IEEE (ieeecolor) Overleaf project."""
import re
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
NAMES = "p1_audit p1_resampling p1_gravity_removal p2_time_series p2_fft_per_class p2_psd_all p2_cwt_per_class p2_envelope_rotation_test p2_feature_ranking p2_feature_scatter calib_noise_floor calib_fan_and_hum p3_confusion_clean p3_final_model_regions p3_window_length p3_robustness_curves p3_robustness_heatmap p3_ablation".split()
NARROW = {"p1_resampling", "p3_final_model_regions", "p3_window_length"}
REPO = "https://github.com/KushajM/SmartphoneVibrationSensor"
L = open(SRC, encoding="utf-8").read().split("\n")

# ---------------------------------------------------------------- blocks
blocks, i = [], 0
while i < len(L):
    ln = L[i]
    if ln.startswith("[TABLE]"):
        rows = []
        i += 1
        while not L[i].startswith("[/TABLE]"):
            rows.append(L[i][4:].split(" || "))
            i += 1
        blocks.append(("table", rows))
    elif ln.startswith("[P:"):
        m = re.match(r"\[P:([^\]]*)\] ?(.*)", ln)
        st, txt = m.group(1), m.group(2)
        # continuation lines (text containing \n)
        while i + 1 < len(L) and not L[i + 1].startswith("[") and not L[i + 1].startswith("  |"):
            i += 1
            txt += "\n" + L[i]
        blocks.append(("p", st, txt))
    elif ln.startswith("[SDT"):
        blocks.append(("toc",))
    i += 1

# ---------------------------------------------------------------- inline text
UNI = {"±": r"\ensuremath{\pm}", "–": "--", "—": "---", "→": r"\ensuremath{\rightarrow}", "·": r"\ensuremath{\cdot}",
       "≈": r"\ensuremath{\approx}", "×": r"\ensuremath{\times}", "µ": r"\ensuremath{\mu}", "−": r"\ensuremath{-}",
       "√": r"\ensuremath{\surd}", "≥": r"\ensuremath{\geq}", "°": r"\ensuremath{^\circ}", "μ": r"\ensuremath{\mu}",
       "σ": r"\ensuremath{\sigma}", "ω": r"\ensuremath{\omega}", "β": r"\ensuremath{\beta}", "Δ": r"\ensuremath{\Delta}",
       "π": r"\ensuremath{\pi}", "Σ": r"\ensuremath{\Sigma}", "≤": r"\ensuremath{\leq}", "…": r"\ldots{}",
       "½": r"\ensuremath{\tfrac12}", "⌈": r"\ensuremath{\lceil}", "⌉": r"\ensuremath{\rceil}",
       "⇒": r"\ensuremath{\Rightarrow}", "←": r"\ensuremath{\leftarrow}", "′": r"\ensuremath{'}",
       "∉": r"\ensuremath{\notin}", "∞": r"\ensuremath{\infty}", "∝": r"\ensuremath{\propto}", "✔": r"\ensuremath{\checkmark}", "’": "'", "‘": "`", "“": "``", "”": "''"}
SUP = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-"))
SUP["²"], SUP["¹"], SUP["³"] = "2", "1", "3"
SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉ₙₖ", "0123456789nk"))

# inline math snippets, matched on the raw text before escaping (longest first)
MATH = [
    ("n = round((tₙ − t₀)/Δt_median)", r"$n = \mathrm{round}\big((t_n - t_0)/\Delta t_\mathrm{median}\big)$"),
    ("S_a = 4·k_B·T·ω₀ / (m·Q)", r"$S_a = 4 k_B T \omega_0/(mQ)$"),
    ("n = n_s(1 − s)", r"$n = n_s(1-s)$"),
    ("dn/n = (2s/(1 + s))·dV/V", r"$\mathrm{d}n/n = \frac{2s}{1+s}\,\mathrm{d}V/V$"),
    ("K·V²·s = fan load c·n²", r"$K V^2 s$ = fan load $c\,n^2$"),
    ("μ(t) = frac(t · (f_s,in − 400 Hz))", r"$\mu(t) = \mathrm{frac}\big(t\,(f_{s,\mathrm{in}} - 400\ \mathrm{Hz})\big)$"),
    ("f_s,in", r"$f_{s,\mathrm{in}}$"), ("f_s/N", r"$f_s/N$"), ("f_s", r"$f_s$"),
    ("σ_f = f_c / 24", r"$\sigma_f = f_c/24$"), ("σ_t", r"$\sigma_t$"), ("σ_f", r"$\sigma_f$"), ("f_c", r"$f_c$"),
    ("S_w + S_p/f", r"$S_w + S_p/f$"), ("SNR(f) = 10·log₁₀(P/F)", r"$\mathrm{SNR}(f) = 10\log_{10}(P/F)$"),
    ("k·f0", r"$k f_0$"), ("3·f0", r"$3 f_0$"), ("f0/3", r"$f_0/3$"), ("2f0/3", r"$2f_0/3$"), ("3f0", r"$3f_0$"),
    ("2s/(1+s)", r"$2s/(1+s)$"), ("cos(ω/2)", r"$\cos(\omega/2)$"),
    ("|H|", r"$|H|$"), ("|W| = A", r"$|W| = A$"), ("1 − (1 + (2/32)⁸)⁻¹", r"$1 - \big(1 + (2/32)^8\big)^{-1}$"),
    ("P(on)", r"$P(\mathrm{on})$"),
    ("m/s^2/sqrt(s)", r"m/s$^2/\sqrt{\mathrm{s}}$"), ("m/s^2", r"m/s$^2$"), ("deg C", r"$^\circ$C"),
]
FEAT = re.compile(r"\b[A-Za-z0-9]+(?:_[A-Za-z0-9]+)+\b")


def esc(s):
    out = []
    for ch in s:
        if ch in "&%$#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch == "<":
            out.append(r"\textless{}")
        elif ch == ">":
            out.append(r"\textgreater{}")
        elif ch == "|":
            out.append(r"\textbar{}")
        else:
            out.append(ch)
    s = "".join(out)
    s = re.sub("[" + "".join(SUP) + "]+", lambda m: r"\ensuremath{^{" + "".join(SUP[c] for c in m.group()) + "}}", s)
    s = re.sub("[" + "".join(SUB) + "]+", lambda m: r"\ensuremath{_{" + "".join(SUB[c] for c in m.group()) + "}}", s)
    s = "".join(UNI.get(c, c) for c in s)
    bad = [c for c in s if ord(c) > 127]
    if bad:
        print("UNMAPPED", set(bad), file=sys.stderr)
    return s


def text(s, math=True):
    """plain text -> LaTeX with inline-math snippets, feature names, URLs and cross-references."""
    ph = []

    def keep(latex):
        ph.append(latex)
        return f"\x00{len(ph) - 1}\x00"

    s = re.sub(r"https?://\S+", lambda m: keep(r"\url{" + m.group().rstrip(".;,") + "}") +
               (m.group()[-1] if m.group()[-1] in ".;," else ""), s)
    if math:
        for a, b in MATH:
            s = s.replace(a, keep(b))
        s = re.sub(r"\bf0\b", lambda m: keep(r"$f_0$"), s)
    # cross references
    s = re.sub(r"Figure (\d+)([a-d]?)", lambda m: keep(rf"Figure~\ref{{fig:{m.group(1)}}}{m.group(2)}"), s)
    s = re.sub(r"Section (\d+(?:\.\d+)?)", lambda m: keep(rf"Section~\ref{{sec:{m.group(1)}}}"), s)
    s = re.sub(r"equation \((\d)\)", lambda m: keep(rf"equation~(\ref{{eq:{m.group(1)}}})"), s)
    s = s.replace("Appendix B", keep(r"Appendix~\ref{app:code}"))
    s = re.sub(r'"([^"\n]*)"', "\x01\\1\x02", s)
    out = esc(s).replace("\x01", "``").replace("\x02", "''")
    # allow line breaks after underscores in long identifiers
    out = out.replace(r"\_", r"\_\allowbreak{}")
    return re.sub("\x00(\\d+)\x00", lambda m: ph[int(m.group(1))], out)


RUN = re.compile(r"\{([bim_^]+)\|(.*?)\}(?=[^}]*(?:\{|$))", re.S)


def runs(s):
    """split '{fl|txt}' runs written by dump_docx.py"""
    parts, pos = [], 0
    for m in re.finditer(r"\{([bim_^]{1,4})\|", s):
        if m.start() < pos:
            continue
        # find matching close brace (runs never nest)
        j = s.find("}", m.end())
        while j != -1 and s[m.end():j].count("{") > s[m.end():j].count("}"):
            j = s.find("}", j + 1)
        if m.start() > pos:
            parts.append(("", s[pos:m.start()]))
        parts.append((m.group(1), s[m.end():j]))
        pos = j + 1
    if pos < len(s):
        parts.append(("", s[pos:]))
    return parts


def inline(s):
    out = ""
    for fl, t in runs(s):
        if "m" in fl:
            body = esc(t).replace("-", "-{}").replace(r"\_", r"\_\allowbreak{}").replace("/", r"/\allowbreak{}")
            r = r"\texttt{" + body + "}"
        else:
            r = text(t)
        if "b" in fl:
            r = r"\textbf{" + r + "}"
        if "i" in fl:
            r = r"\emph{" + r + "}"
        out += r
    return out


# ---------------------------------------------------------------- equations (written by hand)
EQ = {
    "1": r"t_n &\approx t_0 + \frac{n}{f_{s,\mathrm{in}}}, \\ f_{s,\mathrm{in}} &= \frac{1}{\text{slope}}",
    "2": r"H(\mu,\omega) &= 1-\mu+\mu\,e^{-j\omega}, \\ |H|^2 &= 1-2\mu(1-\mu)(1-\cos\omega), \\ \omega &= \frac{2\pi f}{f_s}",
    "3": r"h(d) &= r\,\mathrm{sinc}(r d)\,\frac{I_0\!\left(\beta\sqrt{1-(d/L)^2}\right)}{I_0(\beta)}, \\ L &= \lceil 16/r\rceil \text{ input samples}, \\ \beta &= 8,\quad r=\min\!\left(1,\frac{f_\mathrm{out}}{f_\mathrm{in}}\right)",
    "4": r"&|x_n-\mathrm{med}_n| > 6\cdot\max\!\left(1.4826\,\mathrm{MAD}_n,\ \sigma_\mathrm{record}\right) \\ &\;\Rightarrow\; x_n \leftarrow \mathrm{med}_n",
    "5": r"S(f_0) &= \frac{1}{K'}\sum_{k} \min\!\Big(\max\big(\mathrm{SNR}(k f_0),\,0\big),\ 30\ \mathrm{dB}\Big), \\ & k = 1\ldots4,\quad k f_0 < 0.45\,f_{s,\mathrm{in}}, \\ & k f_0 \notin \{50, 100, 150\}\pm 1\ \mathrm{Hz}",
    "6": r"P(\mathrm{on}) &= \sigma\!\left(\mathbf{w}_A^{\top}\mathbf{z}_A + b_A\right), \\ P(\text{speed } k \mid \mathrm{on}) &= \mathrm{softmax}_k\!\left(\mathbf{W}_B\,\mathbf{z}_B + \mathbf{b}_B\right)",
    "7": r"P(\mathrm{off}) &= 1 - P(\mathrm{on}), \\ P(\text{speed } k) &= P(\mathrm{on})\,P(\text{speed } k \mid \mathrm{on}); \\ & \text{decide off if } P(\mathrm{on}) < 0.5",
}
LONG_EQ = set()

# ---------------------------------------------------------------- tables
FN_EM = 8.0                    # \footnotesize em in ieeecolor [pt]
TEXTW, COLW = 516.0, 252.0     # \textwidth, \columnwidth [pt]
PAD = 1.0                      # 2 x \tabcolsep (4 pt) per column, in em


def tidy(c):
    """Typographic clean-up of generated table text: arrows, +-, units, raw Python dicts, 1e-04 numbers."""
    if "{m|" in c:
        return c

    def fmt_dict(m):
        items = re.findall(r"'(\w+)': (-?[\d.e-]+)", m.group(0))
        return "(" + ", ".join(f"{k} {float(v):.2g}" for k, v in items) + ")"

    c = re.sub(r"\{'\w+': -?[\d.e-]+(?:, '\w+': -?[\d.e-]+)*\}", fmt_dict, c)
    c = c.replace("->", "→").replace("+-", "±").replace("ug/rtHz", "µg/√Hz")
    c = re.sub(r"(\d) us\b", r"\1 µs", c)
    c = re.sub(r"(\d) deg\b", r"\1°", c)
    c = re.sub(r"(\d) C\b", r"\1 °C", c)
    c = re.sub(r"\bsigma\b", "σ", c)
    c = c.replace(" ~ ", " ∝ ")
    c = re.sub(r"(\d|\)) x ", r"\1 × ", c)
    c = c.replace(" .. inf", " to ∞")
    c = re.sub(r"(?<=\d)\s*\.\.\s*(?=[-±+]?\d)", "–", c)

    def sci(m):
        sup = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
        exp = m.group(3).lstrip("0") or "0"
        return m.group(1) + "·10" + (("-" if m.group(2) == "-" else "") + exp).translate(sup)

    c = re.sub(r"(\d+(?:\.\d+)?)e([-+]?)(\d+)\b", sci, c)
    c = re.sub(r"(?<![\w.])-(?=\d)", "−", c)          # minus sign, not hyphen
    return c


def cell(c, header):
    c = c.strip()
    m = re.fullmatch(r"\{b\|(.*)\}", c, re.S)
    if header and m:
        return r"\hspace{0pt}\textbf{" + inline(m.group(1)) + "}"
    return r"\hspace{0pt}" + inline(c) if c else ""


def str_em(s, bold=False, mono=False):
    """rough width of a string in em of the table font (Times / Courier)."""
    if mono:
        return 0.6 * len(s)
    w = 0.0
    for ch in s:
        if ch.isdigit():
            w += 0.5
        elif ch in "mwMW":
            w += 0.78
        elif ch.isupper():
            w += 0.68
        elif ch == " ":
            w += 0.25
        elif ch in "ijlrtf.,:;'!|()[]/-_":
            w += 0.32
        elif ch.isalpha():
            w += 0.47
        else:
            w += 0.6
    return w * (1.08 if bold else 1.0)


def parts(c):
    return [(t, "b" in fl, "m" in fl) for fl, t in runs(c.strip())]


def cell_em(c):
    return sum(str_em(t, b, m) for t, b, m in parts(c))


def word_em(c):
    """widest unbreakable piece: lines break at spaces, after hyphens, underscores and slashes."""
    best = 0.0
    for t, b, m in parts(c):
        for w in re.split(r" |(?<=[-_/])", t):
            best = max(best, str_em(w, b, m))
    return best


NUM = re.compile(r"[−+±]?[\d.,]+\s*%?(?:\s*(?:±|/|–|−|×|,)\s*[−]?[\d.,]+\s*%?)*\s*(?:Hz|dB)?|n/a|none|uncalibrated")


def numeric(cells):
    vals = [re.sub(r"\{[bim]+\|(.*?)\}", r"\1", c.strip()) for c in cells if c.strip()]
    return bool(vals) and sum(bool(NUM.fullmatch(v)) for v in vals) >= 0.7 * len(vals)


def waterfill(dem, wmin, avail):
    """max-min fair split of the available width: narrow columns get what they need, wide ones share the rest."""
    n = len(dem)
    w, free, left = [0.0] * n, list(range(n)), avail
    while free:
        share = left / len(free)
        fixed = [k for k in free if dem[k] <= share]
        if not fixed:
            tot = sum(dem[k] for k in free)
            for k in free:
                w[k] = max(wmin[k], left * dem[k] / tot)
            break
        for k in fixed:
            w[k] = dem[k]
            left -= dem[k]
            free.remove(k)
    if sum(w) < avail:
        w = [x * avail / sum(w) for x in w]
    return w


def pieces(c):
    """widths of the unbreakable pieces of a cell; a line may break between any two pieces."""
    out = []
    for t, b, m in parts(c):
        for w in re.split(r"(?<= )|(?<=[-_/])", t):
            if w:
                out.append(str_em(w, b, m))
    return out


def lines_needed(c, width):
    n, cur = 1, 0.0
    for p in pieces(c):
        if cur and cur + p > width:
            n, cur = n + 1, p
        else:
            cur += p
    return n


def two_line_width(c):
    """narrowest width at which the cell fits on two lines."""
    ps = pieces(c)
    if len(ps) < 2:
        return sum(ps)
    return min(max(sum(ps[:i]), sum(ps[i:])) for i in range(1, len(ps))) * 1.05


def layout(rows):
    """choose one column (table) or both columns (table*) and the width of every column [em]."""
    n = len(rows[0])
    hdr, data = rows[0], rows[1:]
    col = [[(r + [""] * n)[k] for r in data] for k in range(n)]
    natural = [max([cell_em(c) for c in col[k]] + [0.0]) for k in range(n)]
    p80 = [sorted(cell_em(c) for c in col[k])[int(0.8 * (len(col[k]) - 1))] if col[k] else 0.0 for k in range(n)]
    wmin = [max([word_em(c) for c in col[k]] + [word_em(hdr[k])]) * 1.04 for k in range(n)]
    col_av = COLW / FN_EM - n * PAD - 0.4
    txt_av = TEXTW / FN_EM - n * PAD - 0.4
    nat = [max(natural[k] * 1.06, wmin[k]) for k in range(n)]

    def two_line_headers(w, avail):          # widen columns so no header needs more than two lines, if room
        want = [max(w[k], two_line_width(hdr[k])) for k in range(n)]
        return want if sum(want) <= avail else w

    def header_lines(w):
        return max(lines_needed(hdr[k], w[k]) for k in range(n))

    if sum(nat) <= col_av:
        w = two_line_headers(nat, col_av)
        wide = two_line_headers(nat, txt_av)
        if header_lines(w) <= 3 or sum(wide) > txt_av:    # 4+-line headers: use both columns instead
            return "table", w
        return "table*", wide
    if n <= 3 and len(data) <= 25 and sum(wmin) <= col_av and \
            sum(max(p80[k] * 1.06, wmin[k]) for k in range(n)) <= 1.25 * col_av:
        return "table", waterfill(nat, wmin, col_av)
    if sum(nat) <= txt_av:
        w = two_line_headers(nat, txt_av)
        if sum(w) > 0.8 * txt_av:
            w = [x * txt_av / sum(w) for x in w]
        return "table*", w
    # long text: short columns keep their natural width, the long ones share the rest
    dem = [max((natural[k] if natural[k] <= 0.25 * txt_av else p80[k]) * 1.06, wmin[k]) for k in range(n)]
    return "table*", waterfill(dem, wmin, txt_av)


def table(rows, caption=None, label=None):
    rows = [[tidy(c) for c in r] for r in rows]
    n = len(rows[0])
    env, widths = layout(rows)
    spec = ""
    for k in range(n):
        al = r"\centering" if k and numeric([(r + [""] * n)[k] for r in rows[1:]]) else r"\raggedright"
        spec += r">{" + al + r"\arraybackslash}p{%.2fem}" % widths[k]
    wraps = any(cell_em((r + [""] * n)[k]) > widths[k] for r in rows[1:] for k in range(n))
    end = r" \\[1.5pt]" if wraps else r" \\"
    out = [r"\begin{" + env + "}" + ("[!htbp]" if env == "table" else "[!tp]")]
    if caption:
        out.append(r"\caption{" + caption + "}" + (r"\label{" + label + "}" if label else ""))
    out += [r"\centering\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{" + spec + "}", r"\hline",
            " & ".join(cell(c, True) for c in rows[0]) + r" \\ \hline"]
    out += [" & ".join(cell(c, False) for c in (r + [""] * n)[:n]) + end for r in rows[1:]]
    out += [r"\hline", r"\end{tabular}", r"\end{" + env + "}"]
    return "\n".join(out)


# full robustness results: the 22 per-family tables of the docx, merged into four one-column tables
ROB_GROUPS = [
    ("Gaussian-type and drift noise", ["gaussian", "johnson_nyquist", "jn_proofmass", "pink", "brownian", "popcorn"]),
    ("impulsive, multiplicative, quantisation and mechanical disturbances",
     ["salt_pepper", "speckle", "coupling", "quantize", "footsteps", "knocks"]),
    ("tonal interference, orientation, speed shift and dropped samples",
     ["mains_hum", "appliance", "orientation", "speed_shift", "dropouts"]),
    ("lower sampling rates and the calibrated mixes",
     ["lower_fs", "lower_fs_aliased", "mix_same_day", "mix_other_day", "mix_other_phone"]),
]


def rob_tables(rob):
    by = {f: (unit, rows) for f, unit, rows in rob}
    assert sorted(by) == sorted(f for _, fs in ROB_GROUPS for f in fs), sorted(by)
    head = (r"\textbf{Level} & \textbf{R} & \textbf{T1} & \textbf{T2} & \textbf{HLR\_aug} & \textbf{HLR\_aug\_} \\" + "\n"
            + r" & & \textbf{textbook} & \textbf{physics} & \textbf{(unseen)} & \textbf{known} \\ \hline")
    out = []
    for i, (name, fams) in enumerate(ROB_GROUPS, 1):
        lines = [r"\begin{table}[!tbp]",
                 r"\caption{Full robustness results (" + f"{i}/{len(ROB_GROUPS)}" + "): " + name
                 + r". Accuracy (\%, mean $\pm$ sd over 5 repeated 5-fold CVs) at every tested level; "
                 + r"R = level inside the calibrated realistic range.}\label{tab:full" + str(i) + "}",
                 r"\centering\footnotesize\setlength{\tabcolsep}{4.5pt}",
                 r"\begin{tabular}{lccccc}", r"\hline", head]
        for j, f in enumerate(fams):
            unit, rows = by[f]
            if j:
                lines.append(r"\hline")
            lines.append(r"\multicolumn{6}{l}{\rule{0pt}{9pt}\textbf{" + text(f) + "}: " + unit + r"} \\")
            for r in rows:
                lines.append(" & ".join(inline(tidy(c.strip())) for c in r) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        out.append("\n".join(lines))
    return out


ROB_INTRO = (r"Tables~\ref{tab:full1}--\ref{tab:full" + str(len(ROB_GROUPS)) + r"} give the full results: accuracy "
             r"(\%, mean $\pm$ sd over 5 repeated 5-fold CVs) at every tested level of every noise family; R marks the "
             r"levels inside the calibrated realistic range.")

# ---------------------------------------------------------------- figures on landscape pages
LANDSCAPE = {"p3_robustness_curves": ["p3_robustness_curves_part1", "p3_robustness_curves_part2"],
             "p3_robustness_heatmap": ["p3_robustness_heatmap_landscape"]}
land_labels = []


def landscape_float(img, caption, label, cont=False):
    """full-page float, rotated by 90 deg; the page itself is shown in landscape (see \\landscapelabels)."""
    land_labels.append(label)
    return "\n".join([r"\begin{figure*}[p]", r"\centering",
                      r"\rotatebox{90}{\begin{minipage}{0.98\textheight}\centering",
                      r"\includegraphics[width=\linewidth,height=\landscapeimgheight,keepaspectratio]{figures/" + img + "}",
                      (r"\addtocounter{figure}{-1}" if cont else "") + r"\caption{" + caption + r"}\label{" + label + "}",
                      r"\end{minipage}}", r"\end{figure*}"])


# ---------------------------------------------------------------- limitations / future work as prose
LIMITS = r"""All recordings come from one fan, one phone, one mount and one evening, so cross-validation measures repeatability within that session. Day-to-day supply-voltage changes are the main threat (Section~\ref{sec:7.3}), and the calibrated other-day mix ($\approx$~88~\%) is the best available estimate of cross-day performance. Some disturbances are calibrated rather than measured: the footstep and knock amplitudes rest on literature anchors, and the re-taping effects and the appliance amplitude are assumptions; all are flagged in Section~\ref{sec:5.5}.

Feature calibration used all 80 trials to set the realistic noise ranges (not to fit the classifier). This is a mild form of using test data for design; the leave-one-group-out protocol is the safeguard for the classifier itself. The rotation hypothesis rests on spectral evidence, so a blade count and an independent rpm measurement are still needed. Finally, the demo predictions are in-sample; the reported accuracy is always the cross-validated one.
"""

FUTURE = r"""The first step is a second recording session on another day, after re-taping and with the mains voltage noted, on which the frozen model is tested. Counting the blades and measuring the rpm with a slow-motion video would close the rotation question, and a few trials recorded with a fridge or AC running and with footsteps would calibrate the two uncalibrated disturbance amplitudes. For deployment, a per-deployment calibration (one recording per speed) would absorb supply-voltage drift, and a reject option (``uncertain'') could be returned when $P(\mathrm{on})$ or the speed probability is low. Finally, the classifier could run live on a Phyphox remote stream, since 0.5--1~s windows are sufficient (Section~\ref{sec:6.5}).
"""

# ---------------------------------------------------------------- walk
body, abstract, lists, refs, rob = [], [], [], [], []
software = None
state = "front"
fig_n = 0
pending_img = None
pending_caption = None
code_lines = []
skip = False


def flush_list():
    global lists
    if lists:
        body.append("\\begin{itemize}\n" + "\n".join(r"\item " + x for x in lists) + "\n\\end{itemize}")
        lists = []


def flush_code():
    global code_lines
    if code_lines:
        body.append("\\begin{Verbatim}[fontsize=\\tiny,breaklines=true,frame=single]\n" + "\n".join(code_lines) + "\n\\end{Verbatim}")
        code_lines = []


k = 0
while k < len(blocks):
    b = blocks[k]
    if b[0] == "toc":
        k += 1
        continue
    if b[0] == "table":
        flush_list(); flush_code()
        if state == "appD":
            rows = [r[:] for r in b[1]]
            fam = re.fullmatch(r"\{b\|(.*)\}", rows[0][0].strip()).group(1)
            m = re.match(r"(\S+) \((.*)\)$", fam)
            rob.append((m.group(1), text(tidy(m.group(2))), rows[1:]))
        elif state == "appA":
            if pending_caption and pending_caption[0].startswith("Software environment"):
                software = b[1]
        elif not skip:
            body.append(table(b[1], *(pending_caption or (None, None))))
        pending_caption = None
        k += 1
        continue
    _, st, t = b
    t_strip = t.strip()
    if st.startswith("Heading"):
        flush_list(); flush_code()
        lvl = int(st[7])
        title = re.sub(r"\{[bim]+\|(.*?)\}", r"\1", t_strip)
        if lvl == 1:
            skip = False
            if title == "Abstract":
                state = "abstract"
            elif title == "References":
                state = "refs"
            elif title.startswith("Appendix A"):
                state = "appA"
            elif title.startswith(("Appendix B", "Appendix C")):
                state = "appBC"; skip = True
            elif title.startswith("Appendix D"):
                state = "appD"
            else:
                state = "body"
                m = re.match(r"(\d+)\. (.*)", title)
                body.append(rf"\section{{{text(m.group(2), math=False).strip()}}}\label{{sec:{m.group(1)}}}")
        elif state == "body":
            if lvl == 2:
                m = re.match(r"(\d+\.\d+) (.*)", title)
                body.append(rf"\subsection{{{text(m.group(2), math=False).strip()}}}\label{{sec:{m.group(1)}}}")
            else:
                body.append(rf"\subsubsection{{{text(title)}}}")
        k += 1
        continue
    if state == "front" or skip:
        k += 1
        continue
    if state == "abstract":
        if t_strip:
            abstract.append(inline(t_strip))
        k += 1
        continue
    # caption paragraphs
    mcap = re.match(r"\{b\|(Table|Figure) (\d+)\.\}\s*(.*)", t_strip, re.S)
    if mcap:
        flush_list(); flush_code()
        kind, num, rest = mcap.groups()
        if kind == "Table":
            pending_caption = (inline(rest), f"tab:{num}")
        else:
            if pending_img in LANDSCAPE:
                imgs = LANDSCAPE[pending_img]
                if len(imgs) == 2:
                    body.append(landscape_float(imgs[0], inline(rest) + " (Part 1 of 2, continued on the next page.)", "fig:" + num))
                    body.append(landscape_float(imgs[1], "(Continued) Accuracy vs noise level, part 2 of 2.", "fig:" + num + "b", cont=True))
                else:
                    body.append(landscape_float(imgs[0], inline(rest), "fig:" + num))
            else:
                # narrow plots fit one column; multi-panel plots span both columns
                if pending_img in NARROW:
                    env, opts, inc = "figure", "[!t]", "width=\\columnwidth"
                else:
                    env, opts, inc = "figure*", "[!tp]", "width=\\textwidth,height=0.8\\textheight,keepaspectratio"
                body.append("\\begin{" + env + "}" + opts + "\n\\centering\n\\includegraphics[" + inc + "]{figures/"
                            + pending_img + "}\n\\caption{" + inline(rest) + "}\\label{fig:" + num + "}\n\\end{" + env + "}")
            pending_img = None
        k += 1
        continue
    if state in ("appA", "appD"):          # appendix prose is not carried over
        k += 1
        continue
    mimg = re.fullmatch(r"<IMG media/(image\d+)\.png>", t_strip)
    if mimg:
        pending_img = NAMES[int(mimg.group(1)[5:]) - 1]
        k += 1
        continue
    # equations: paragraph ending with (n)
    meq = re.search(r"\((\d)\)\s*$", t_strip)
    if meq and state == "body" and ("=" in t_strip or "⇒" in t_strip or "≈" in t_strip) and len(t_strip) < 260:
        flush_list(); flush_code()
        n = meq.group(1)
        eq = EQ[n]
        if "&" in eq:
            body.append("\\begin{equation}\\label{eq:" + n + "}\n\\begin{split}\n" + eq + "\n\\end{split}\n\\end{equation}")
        else:
            body.append("\\begin{equation}\\label{eq:" + n + "}\n" + eq + "\n\\end{equation}")
        k += 1
        continue
    # monospace-only paragraphs = code/console block
    mcode = re.fullmatch(r"\{m\|(.*)\}", t, re.S)
    if mcode and state == "body" and not st.startswith("List"):
        flush_list()
        code_lines.append(mcode.group(1).rstrip())
        k += 1
        continue
    flush_code()
    if st.startswith("ListParagraph"):
        lists.append(inline(t_strip))
        k += 1
        continue
    flush_list()
    if not t_strip:
        k += 1
        continue
    if state == "refs":
        mref = re.match(r"\[(\d+)\] (.*)", t_strip)
        refs.append(r"\bibitem{ref" + mref.group(1) + "} " + inline(mref.group(2)))
        k += 1
        continue
    if "The full MATLAB code is listed in Appendix B." in t_strip:
        t_strip = t_strip.replace("The full MATLAB code is listed in Appendix B.",
                                  "The complete MATLAB and Python code is in the project's GitHub repository, " + REPO + ".")
    body.append(inline(t_strip) + "\n")
    k += 1
flush_list(); flush_code()
assert software and len(rob) == 22 and refs, (software is None, len(rob), len(refs))

# ---------------------------------------------------------------- assemble
body_tex = "\n".join(body)
for lab, prose in (("9.2", LIMITS), ("9.3", FUTURE)):
    body_tex, nsub = re.subn(r"(\\label\{sec:" + re.escape(lab) + r"\}\n)\\begin\{itemize\}.*?\\end\{itemize\}",
                             lambda m: m.group(1) + prose, body_tex, count=1, flags=re.S)
    assert nsub == 1, lab
anchor = r"\subsection{Interpretation}\label{sec:7.3}"
assert body_tex.count(anchor) == 1
body_tex = body_tex.replace(anchor, ROB_INTRO + "\n\n" + "\n\n".join(rob_tables(rob)) + "\n\n" + anchor)
# IEEE drop cap on the first paragraph of the introduction
body_tex = re.sub(r"(\\label\{sec:1\}\n(?:\\subsection[^\n]*\n)?)([A-Z])([a-z]*)",
                  lambda m: m.group(1) + r"\IEEEPARstart{" + m.group(2) + "}{" + m.group(3) + "}", body_tex, count=1)

# heading, sentence and table in one unbreakable block so the heading never ends a column on its own
sw = [r"\FloatBarrier", r"\section*{Software Environment}",
      r"\noindent\begin{minipage}{\columnwidth}",
      r"\parindent=1em\indent All results were produced with the software below; the code, raw recordings and results "
      r"are at \url{" + REPO + "}.\par\medskip",
      r"\centering\footnotesize", r"\begin{tabular}{ll}", r"\hline",
      " & ".join(cell(c, True) for c in software[0]) + r" \\ \hline"]
sw += [" & ".join(cell(c, False) for c in r) + r" \\" for r in software[1:]]
sw += [r"\hline", r"\end{tabular}", r"\end{minipage}", r"\medskip"]
bib = [r"\begin{thebibliography}{00}"] + refs + [r"\end{thebibliography}"]
body_tex += "\n\n" + "\n".join(sw) + "\n\n" + "\n".join(bib) + "\n"

title = "Smartphone-Based Vibration Signature Analysis"
subtitle = text("Classifying the operating state of a wall fan from a phone's MEMS accelerometer: signal processing, calibrated noise robustness and a noise-robust ML classifier")

pre = r"""%% main.tex -- Smartphone-Based Vibration Signature Analysis (IEEE ieeecolor template)
\documentclass[journal,twoside,web]{ieeecolor}
\usepackage{generic}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{array}
\usepackage{fvextra}
\usepackage{placeins}
\usepackage{atbegshi,refcount}
\usepackage{url}
\usepackage{hyperref}
\hypersetup{hidelinks=true}
\def\UrlFont{\rm}
\urlstyle{rm}
\renewcommand{\arraystretch}{1.15}
\setcounter{dbltopnumber}{3}
\setcounter{topnumber}{4}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{6}
\renewcommand{\dbltopfraction}{0.85}
\renewcommand{\topfraction}{0.9}
\renewcommand{\bottomfraction}{0.6}
\renewcommand{\textfraction}{0.07}
\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\dblfloatpagefraction}{0.75}
\sloppy
% course report, not an IEEE submission: course name instead of the journal name in the headers
\def\journalname{PROJECT COMPUTATION COURSE, JADAVPUR UNIVERSITY}
\markboth{\hskip25pc PROJECT COMPUTATION COURSE, JADAVPUR UNIVERSITY}
{Smartphone-Based Vibration Signature Analysis}
% Landscape figure pages: the figure is rotated by 90 deg on its own float page, and the PDF page is
% flagged /Rotate 90 so viewers show it in landscape. The page numbers come from the figure labels
% (correct from the second LaTeX run on, which Overleaf does automatically).
\newlength{\landscapeimgheight}
\AtBeginDocument{\setlength{\landscapeimgheight}{0.9\textwidth}}
\newcommand*{\landscapelabels}{""" + ",".join(land_labels) + r"""}
\makeatletter
\AtBeginDocument{\ifdefined\pdfpageattr\edef\ls@attr{\the\pdfpageattr}\fi}
\AtBeginShipout{\ifdefined\pdfpageattr
  \global\pdfpageattr\expandafter{\ls@attr}%
  \@for\ls@lab:=\landscapelabels\do{%
    \ifnum\getpagerefnumber{\ls@lab}=\value{page}\relax
      \global\pdfpageattr\expandafter{\ls@attr/Rotate 90}\fi}%
\fi}
\makeatother

\begin{document}
\title{""" + title + r"""}
%% Add author names here, e.g. \author{First A. Author, Second B. Author \thanks{...}}
\author{Department of Electrical Engineering, Jadavpur University, Kolkata, India
\thanks{Code, raw recordings and all results: \url{""" + REPO + r"""}.}}

\maketitle

\begin{abstract}
""" + r"\textit{" + subtitle + r"}\par" + "\n\n" + "\n\n".join(abstract) + r"""
\end{abstract}

\begin{IEEEkeywords}
Harmonic comb, logistic regression, MEMS accelerometer, noise robustness, vibration signature analysis.
\end{IEEEkeywords}

"""

post = r"""
\end{document}
"""
open(OUT, "w", encoding="utf-8").write(pre + body_tex + post)
print("landscape:", land_labels, "robustness families:", len(rob), "refs:", len(refs))
