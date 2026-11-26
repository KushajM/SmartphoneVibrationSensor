"""Convert the dumped Final_Report.docx (user.txt) into a single main.tex for the spr-sola Overleaf project."""
import re
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
NAMES = "p1_audit p1_resampling p1_gravity_removal p2_time_series p2_fft_per_class p2_psd_all p2_cwt_per_class p2_envelope_rotation_test p2_feature_ranking p2_feature_scatter calib_noise_floor calib_fan_and_hum p3_confusion_clean p3_final_model_regions p3_window_length p3_robustness_curves p3_robustness_heatmap p3_ablation".split()
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
       "∉": r"\ensuremath{\notin}", "✔": r"\ensuremath{\checkmark}", "’": "'", "‘": "`", "“": "``", "”": "''"}
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
    "1": r"t_n \approx t_0 + \frac{n}{f_{s,\mathrm{in}}}, \qquad f_{s,\mathrm{in}} = \frac{1}{\text{slope}}",
    "2": r"H(\mu,\omega) = 1-\mu+\mu\,e^{-j\omega}, \qquad |H|^2 = 1-2\mu(1-\mu)(1-\cos\omega), \qquad \omega = \frac{2\pi f}{f_s}",
    "3": r"h(d) = r\,\mathrm{sinc}(r d)\,\frac{I_0\!\left(\beta\sqrt{1-(d/L)^2}\right)}{I_0(\beta)}, \qquad L=\lceil 16/r\rceil \text{ input samples},\ \beta=8,\ r=\min\!\left(1,\frac{f_\mathrm{out}}{f_\mathrm{in}}\right)",
    "4": r"|x_n-\mathrm{med}_n| > 6\cdot\max\!\left(1.4826\,\mathrm{MAD}_n,\ \sigma_\mathrm{record}\right) \;\Rightarrow\; x_n \leftarrow \mathrm{med}_n",
    "5": r"S(f_0) = \frac{1}{K'}\sum_{k} \min\!\Big(\max\big(\mathrm{SNR}(k f_0),\,0\big),\ 30\ \mathrm{dB}\Big), \quad k = 1\ldots4,\ \ k f_0 < 0.45\,f_{s,\mathrm{in}},\ \ k f_0 \notin \{50, 100, 150\}\pm 1\ \mathrm{Hz}",
    "6": r"P(\mathrm{on}) &= \sigma\!\left(\mathbf{w}_A^{\top}\mathbf{z}_A + b_A\right), \\ P(\text{speed } k \mid \mathrm{on}) &= \mathrm{softmax}_k\!\left(\mathbf{W}_B\,\mathbf{z}_B + \mathbf{b}_B\right)",
    "7": r"P(\mathrm{off}) &= 1 - P(\mathrm{on}), \qquad P(\text{speed } k) = P(\mathrm{on})\,P(\text{speed } k \mid \mathrm{on}); \\ & \text{decide off if } P(\mathrm{on}) < 0.5",
}
LONG_EQ = {"3", "5"}

# ---------------------------------------------------------------- tables
def cell(c, header):
    c = c.strip()
    m = re.fullmatch(r"\{b\|(.*)\}", c, re.S)
    if header and m:
        return r"\hspace{0pt}\textbf{" + inline(m.group(1)) + "}"
    return r"\hspace{0pt}" + inline(c) if c else ""


def plain_len(c):
    return len(re.sub(r"\{[bim_^]+\||\}", "", c))


def table(rows, caption=None, label=None, small=False):
    n = len(rows[0])
    widths = [max(plain_len(r[k]) for r in rows[1:]) if len(rows) > 1 else 10 for k in range(n)]
    hdr = [plain_len(c) for c in rows[0]]
    total = sum(max(w, h) for w, h in zip(widths, hdr))
    size = r"\footnotesize" if (n >= 5 or small or total > 120) else r"\small"
    if small:
        spec = "lc" + "c" * (n - 2)
        env = "longtable"
        open_ = r"\begin{longtable}{" + spec + "}"
    elif total <= 62 and n <= 4:
        spec = "l" * n
        env = "longtable"
        open_ = r"\begin{longtable}{" + spec + "}"
    else:
        cols = []
        for k in range(n):
            w = widths[k]
            if n >= 5:
                cols.append(r">{\raggedright\arraybackslash}X" if k == 0 else r">{\centering\arraybackslash}X")
            elif w > 22:
                cols.append(r">{\raggedright\arraybackslash}X")
            else:
                cols.append(r">{\raggedright\arraybackslash}p{%.1fem}" % min(max(max(w, 6) * 0.55, 3.5), 11))
        if not any("X" in c for c in cols):          # xltabular needs at least one flexible column
            cols = [r">{\raggedright\arraybackslash}X"] * n
        env = "xltabular"
        open_ = r"\begin{xltabular}{\textwidth}{" + "".join(cols) + "}"
    out = ["{" + size, r"\setlength{\tabcolsep}{4pt}", open_]
    if caption:
        out.append(r"\caption{" + caption + "}" + (r"\label{" + label + "}" if label else "") + r"\\")
    out.append(r"\hline")
    head = " & ".join(cell(c, True) for c in rows[0]) + r" \\ \hline"
    out += [head, r"\endfirsthead", r"\hline", head, r"\endhead", r"\hline", r"\endfoot"]
    for r in rows[1:]:
        r = (r + [""] * n)[:n]
        out.append(" & ".join(cell(c, False) for c in r) + r" \\")
    out.append(r"\hline")
    out.append(r"\end{" + env + "}}")
    return "\n".join(out)


# ---------------------------------------------------------------- walk
body = []
abstract = []
lists = []
state = "front"
fig_n = 0
pending_img = None
pending_caption = None
code_lines = []
skip = False  # skip appendix B/C code listings
files_table = []


def flush_list():
    global lists
    if lists:
        body.append("\\begin{itemize}\n" + "\n".join(r"\item " + x for x in lists) + "\n\\end{itemize}")
        lists = []


def flush_code():
    global code_lines
    if code_lines:
        body.append("\\begin{Verbatim}[fontsize=\\scriptsize,breaklines=true,frame=single]\n" + "\n".join(code_lines) + "\n\\end{Verbatim}")
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
            # full robustness tables: family + unit become the caption, first header cell becomes "Level"
            rows = [r[:] for r in b[1]]
            fam = re.fullmatch(r"\{b\|(.*)\}", rows[0][0].strip()).group(1)
            rows[0][0] = "{b|Level}"
            m = re.match(r"(\S+) \((.*)\)$", fam)
            cap = r"\texttt{" + esc(m.group(1)) + "}: " + text(m.group(2)) if m else text(fam)
            body.append(table(rows, cap, "tab:rob:" + (m.group(1) if m else fam), small=True))
        elif not skip:
            body.append(table(b[1], pending_caption[0] if pending_caption else None,
                              pending_caption[1] if pending_caption else None,
                              small=(state == "appD")))
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
                body.append(r"\section*{References}" + "\n" + r"\addcontentsline{toc}{section}{References}")
            elif title.startswith("Appendix A"):
                state = "appA"
                body.append("\\appendix\n\\section{Reproducibility}\\label{app:repro}")
            elif title.startswith("Appendix B"):
                state = "appB"; skip = True
            elif title.startswith("Appendix C"):
                state = "appC"; skip = True
            elif title.startswith("Appendix D"):
                state = "appD"
                body.append(r"\section{Full robustness results}\label{app:robust}")
            else:
                state = "body"
                m = re.match(r"(\d+)\. (.*)", title)
                body.append(rf"\section{{{text(m.group(2), math=False).strip()}}}\label{{sec:{m.group(1)}}}")
        elif lvl == 2:
            m = re.match(r"(\d+\.\d+) (.*)", title)
            body.append(rf"\subsection{{{text(m.group(2), math=False).strip()}}}\label{{sec:{m.group(1)}}}")
        else:
            if state in ("appB", "appC"):
                m = re.match(r"(.*?)\s+\((\S+), (\d+) lines\)", title)
                if m and "build_report" not in m.group(2):
                    files_table.append((m.group(2), m.group(1), m.group(3)))
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
            if pending_img in ("p3_robustness_curves", "p3_robustness_heatmap"):
                # wide multi-panel figures: rotate onto their own landscape page so they stay legible
                body.append("\\begin{sidewaysfigure}\n\\centering\n\\includegraphics[width=\\textheight,height=0.9\\textwidth,keepaspectratio]{figures/"
                            + pending_img + "}\n\\caption{" + inline(rest) + "}\\label{fig:" + num + "}\n\\end{sidewaysfigure}")
            else:
                body.append("\\begin{figure}[tbp]\n\\centering\n\\includegraphics[width=\\textwidth,height=0.72\\textheight,keepaspectratio]{figures/"
                            + pending_img + "}\n\\caption{" + inline(rest) + "}\\label{fig:" + num + "}\n\\end{figure}")
            pending_img = None
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
        if n in LONG_EQ:
            body.append("\\begin{equation}\\label{eq:" + n + "}\n\\begin{split}\n" +
                        eq.replace(r", \qquad L", r", \\ & L").replace(r", \quad k", r", \\ & k").replace("h(d) =", "h(d) &=").replace("S(f_0) =", "S(f_0) &=")
                        + "\n\\end{split}\n\\end{equation}")
        elif "&" in eq:
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
        body.append(r"\noindent\hangindent=1.8em\hangafter=1 [" + mref.group(1) + "]~" + inline(mref.group(2)) + "\n")
        k += 1
        continue
    # paragraph text edits requested for the LaTeX version
    if "The full MATLAB code is listed in" in t_strip:
        t_strip = t_strip.replace("The full MATLAB code is listed in Appendix B.",
                                  "The complete MATLAB and Python code is in the project's GitHub repository (Appendix B).")
    body.append(inline(t_strip) + "\n")
    k += 1
flush_list(); flush_code()

# ---------------------------------------------------------------- code appendix (GitHub link instead of listings)
code_app = [r"\section{Code and data}\label{app:code}",
            r"All code, the 80 raw recordings and every generated table, figure and model file are in the GitHub repository",
            r"\begin{center}\url{" + REPO + r"}\end{center}",
            r"\noindent The MATLAB port (written for R2018a with the Signal Processing Toolbox) was verified against the Python implementation (Section~\ref{sec:8.2}). Table~\ref{tab:code} lists the source files.",
            ""]
rows = [["{b|File}", "{b|Purpose}", "{b|Lines}"]]
for f, purpose, n in files_table:
    rows.append([f, purpose, n])
tab = table(rows, "Source files in the repository (links open the file on GitHub).", "tab:code")
# hyperlink each file name
for f, _, _ in files_table:
    tab = tab.replace(text(f), r"\href{" + REPO + "/blob/main/" + f + r"}{\texttt{" + text(f) + "}}", 1)
code_app.append(tab)

# Appendix A: drop the Word-report build command row
body_tex = "\n".join(body)
body_tex = re.sub(r"\n[^\n]*build\\_\\allowbreak\{\}report[^\n]*\\\\", "", body_tex)
# put the code appendix where appendix B was (after A, before D)
body_tex = body_tex.replace(r"\section{Full robustness results}", "\n".join(code_app) + "\n\n" + r"\section{Full robustness results}")

title = "Smartphone-Based Vibration Signature Analysis"
subtitle = text("Classifying the operating state of a wall fan from a phone's MEMS accelerometer: signal processing, calibrated noise robustness and a noise-robust ML classifier")

pre = r"""%% main.tex -- Smartphone-Based Vibration Signature Analysis (spr-sola class)
\documentclass[namedreferences,hyperref,optionalrh,noid]{spr-sola}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{array,longtable,tabularx,xltabular}
\usepackage{url}
\usepackage{fvextra}
\usepackage{rotating}
\usepackage{xcolor}
\def\UrlFont{\rm}
\urlstyle{rm}
\providecommand{\contentsname}{Contents}
\setcounter{tocdepth}{2}
\renewcommand{\arraystretch}{1.15}
\setlength{\LTpre}{6pt}\setlength{\LTpost}{6pt}
\setlength{\parskip}{3pt plus 1pt}
\sloppy
% course report, not a journal submission: replace the journal name/DOI header and drop the copyright line
\makeatletter
\def\@jname{Project Computation Course}
\def\@doi{Department of Electrical Engineering, Jadavpur University}
\def\@copyright{}
\makeatother

\begin{document}

\begin{frontmatter}
\title{""" + title + r"""}
%% \author[addressref=aff1,email={}]{\inits{}\fnm{}~\snm{}}
%% \address[id=aff1]{Department of Electrical Engineering, Jadavpur University, Kolkata, India}
\runningauthor{Project Computation Course, Jadavpur University}
\runningtitle{Smartphone-Based Vibration Signature Analysis}

\begin{abstract}
""" + r"\textit{" + subtitle + r"}\par\medskip" + "\n\n" + "\n\n".join(abstract) + r"""
\end{abstract}

\begin{keyword}
\kwd{MEMS accelerometer}
\kwd{Vibration signature analysis}
\kwd{Harmonic comb}
\kwd{Noise robustness}
\kwd{Logistic regression}
\end{keyword}
\end{frontmatter}

\tableofcontents
\clearpage

"""

post = r"""
\begin{codeavailability}
Python pipeline, MATLAB port, raw recordings and all results: \url{""" + REPO + r"""}.
\end{codeavailability}

\end{document}
"""
open(OUT, "w", encoding="utf-8").write(pre + body_tex + post)
print("figures:", fig_n, "files:", len(files_table))
