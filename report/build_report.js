// Builds report/Final_Report.docx from report/report_data.json, results/figures/*.png and the
// source code.  Run:  node report/build_report.js   (needs the npm package "docx")
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, ShadingType, ImageRun, PageBreak, TableOfContents, Footer, Header, PageNumber,
  LevelFormat, BorderStyle,
} = require("docx");

const ROOT = path.resolve(__dirname, "..");
const D = JSON.parse(fs.readFileSync(path.join(ROOT, "report", "report_data.json"), "utf8"));
const FIG = (f) => path.join(ROOT, "results", "figures", f);
const CONTENT_W = 9638;              // A4 (11906) minus 2 x 1134 DXA margins
const FONT = "Calibri";
const MONO = "Consolas";

// ------------------------------------------------------------------ formatting helpers
const f = (x, d = 2) => (x === null || x === undefined || Number.isNaN(x) ? "n/a" : Number(x).toFixed(d));
const pct = (x, d = 1) => (x === null || x === undefined ? "n/a" : (100 * x).toFixed(d) + " %");
const pm = (m, s, d = 1, k = 100) => (m === null || m === undefined ? "n/a" : `${(k * m).toFixed(d)} ± ${(k * s).toFixed(d)}`);
const sci = (x, d = 2) => Number(x).toExponential(d);
const MODEL_DESC = {
  T1_textbook: "Threshold rule (no ML): z-axis RMS -> off/on; largest FFT peak -> speed",
  T2_physics: "Threshold rule (no ML): comb SNR with CFAR threshold -> off/on; comb f0 -> speed",
  kNN_clean: "k-nearest neighbours (k = 5), trained on real trials only",
  kNN_aug: "k-nearest neighbours, trained on real + noise-augmented trials",
  LR_clean: "Multinomial logistic regression, real trials only",
  LR_aug: "Multinomial logistic regression, real + noise-augmented trials",
  HLR_clean: "Two-stage logistic regression, real trials only",
  HLR_aug: "Two-stage logistic regression, real + noise-augmented trials (FINAL)",
  HLR_aug_known: "Final model when the tested noise type was in its training augmentation",
};

// inline markup: **bold**, *italic*, `code`
function runs(text, base = {}) {
  const out = [];
  const parts = String(text).split(/(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g).filter((s) => s.length);
  for (const p of parts) {
    if (p.startsWith("**")) out.push(new TextRun({ text: p.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (p.startsWith("`")) out.push(new TextRun({ text: p.slice(1, -1), font: MONO, size: base.size ? base.size - 2 : 19, ...base, ...{ font: MONO } }));
    else if (p.startsWith("*")) out.push(new TextRun({ text: p.slice(1, -1), italics: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: p, font: FONT, ...base }));
  }
  return out;
}

const body = [];
const P = (text, opts = {}) => body.push(new Paragraph({ children: runs(text, opts.run || {}), spacing: { after: 120, line: 276 }, alignment: opts.align || AlignmentType.JUSTIFIED, ...opts.para }));
const H1 = (t) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text: t })], pageBreakBefore: true }));
const H2 = (t) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text: t })] }));
const H3 = (t) => body.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: t })] }));
const BUL = (items, ref = "bullets") => items.forEach((t) => body.push(new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60, line: 264 } })));
const EQ = (t, label) => body.push(new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 },
  children: [new TextRun({ text: t, font: "Cambria Math", size: 22, italics: false }), ...(label ? [new TextRun({ text: `        (${label})`, font: FONT, size: 20 })] : [])],
}));
const NOTE = (title, lines) => {
  body.push(new Paragraph({
    spacing: { before: 120, after: 60 }, shading: { type: ShadingType.CLEAR, fill: "EAF1FB" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: "2F5597", space: 6 } },
    children: [new TextRun({ text: title, bold: true, font: FONT, color: "1F3864" })],
  }));
  lines.forEach((l) => body.push(new Paragraph({
    spacing: { after: 40, line: 264 }, shading: { type: ShadingType.CLEAR, fill: "EAF1FB" },
    border: { left: { style: BorderStyle.SINGLE, size: 18, color: "2F5597", space: 6 } }, children: runs(l),
  })));
  body.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
};

let figN = 0, tabN = 0;
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function FIGURE(file, caption, frac = 1.0) {
  const p = FIG(file);
  const { w, h } = pngSize(p);
  const maxW = 640 * frac;
  const W = Math.min(maxW, w), H = Math.round((h * W) / w);
  figN += 1;
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 40 }, keepNext: true,
    children: [new ImageRun({ type: "png", data: fs.readFileSync(p), transformation: { width: W, height: H },
      altText: { title: `Figure ${figN}`, description: caption, name: file } })] }));
  body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: runs(`**Figure ${figN}.** ${caption}`, { size: 18, italics: false }) }));
  return figN;
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6" };
const borders = { top: border, bottom: border, left: border, right: border };
function TABLE(headers, rows, widths, caption, opts = {}) {
  const total = widths.reduce((a, b) => a + b, 0);
  const scale = CONTENT_W / total;
  const W = widths.map((w) => Math.floor(w * scale));
  W[W.length - 1] += CONTENT_W - W.reduce((a, b) => a + b, 0);
  const size = opts.size || 17;
  tabN += 1;
  if (caption) body.push(new Paragraph({ spacing: { before: 160, after: 60 }, keepNext: true, children: runs(`**Table ${tabN}.** ${caption}`, { size: 18 }) }));
  const cell = (t, i, head) => new TableCell({
    width: { size: W[i], type: WidthType.DXA }, borders,
    shading: head ? { type: ShadingType.CLEAR, fill: "D9E2F3", color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ spacing: { after: 0, line: 240 }, alignment: i === 0 || opts.left ? AlignmentType.LEFT : AlignmentType.CENTER,
      children: runs(String(t), { size, bold: head ? true : undefined }) })],
  });
  body.push(new Table({
    width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: W,
    rows: [new TableRow({ tableHeader: true, children: headers.map((h, i) => cell(h, i, true)) }),
      ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((t, i) => cell(t, i, false)) }))],
  }));
  body.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
  return tabN;
}
function CODE(file, title) {
  const src = fs.readFileSync(path.join(ROOT, file), "utf8").replace(/\r/g, "").replace(/\t/g, "    ").split("\n");
  if (src[src.length - 1] === "") src.pop();
  body.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun({ text: `${title}  (${file}, ${src.length} lines)` })] }));
  src.forEach((line) => body.push(new Paragraph({
    spacing: { before: 0, after: 0, line: 200, lineRule: "exact" }, shading: { type: ShadingType.CLEAR, fill: "F4F4F4" },
    children: [new TextRun({ text: line.length ? line : " ", font: MONO, size: 13 })],
  })));
  body.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
}

// ------------------------------------------------------------------ derived numbers
const p1 = D.p1, p2 = D.p2, p3 = D.p3, ev = D.cal_evidence;
const cv = Object.fromEntries(D.clean_cv.map((r) => [r.model, r]));
const RS = (fam, m) => D.rob_summary.find((r) => r.family === fam && r.model === m);
const freq = Object.fromEntries(D.freq.map((r) => [r.cls, r]));
const M = D.model;
const A = M.stage_a, B = M.stage_b;
const nStar = A.scaler_mean[0] - (A.intercept[0] * A.scaler_scale[0]) / A.coef[0][0];
const bnd = (i, j) => B.scaler_mean[0] + (-(B.intercept[i] - B.intercept[j]) / (B.coef[i][0] - B.coef[j][0])) * B.scaler_scale[0];
const b12 = bnd(0, 1), b23 = bnd(1, 2);
const sel = (m) => D.selection.filter((r) => r.model === m);
const selMean = (m, cfg) => { const v = sel(m).filter((r) => r.config === cfg).map((r) => r.acc); return v.reduce((a, b) => a + b, 0) / v.length; };
const CFG = ["all features, mean accuracy", "all features, worst-group", "invariant, mean accuracy", "invariant, worst-group (adopted)"];
const hk = (fam) => RS(fam, "HLR_aug_known");
const hu = (fam) => RS(fam, "HLR_aug");
const t2 = (fam) => RS(fam, "T2_physics");
const t1 = (fam) => RS(fam, "T1_textbook");
const ds = Object.fromEntries(ev.datasheets.map((d) => [d.nd_ug, d]));

// =================================================================== TITLE PAGE
body.push(new Paragraph({ spacing: { before: 2400, after: 240 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "Smartphone-Based Vibration Signature Analysis", font: FONT, size: 48, bold: true, color: "1F3864" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 480 },
  children: [new TextRun({ text: "Classifying the operating state of a wall fan from a phone's MEMS accelerometer — signal processing, calibrated noise robustness and a noise-robust ML classifier", font: FONT, size: 26, color: "404040" })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Final report — Project Computation Course, Department of Electrical Engineering, Jadavpur University", font: FONT, size: 22 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: [new TextRun({ text: "Student: ______________________     Roll no.: ______________", font: FONT, size: 22 })] }));
body.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 1200 }, children: [new TextRun({ text: "5 November 2026", font: FONT, size: 22 })] }));
NOTE("At a glance", [
  `**Data:** 80 recordings (20 × off / speed1 / speed2 / speed3), Motorola Edge 60 Fusion, TDK ICM-45621 accelerometer, ${f(p1.fs_fit_mean, 3)} Hz, one ${f(p1.session_minutes, 0)}-minute session.`,
  `**Signal:** a harmonic comb k·f0 with f0 = ${f(freq.speed1.f0_mean)} ± ${f(freq.speed1.f0_sd)}, ${f(freq.speed2.f0_mean)} ± ${f(freq.speed2.f0_sd)}, ${f(freq.speed3.f0_mean)} ± ${f(freq.speed3.f0_sd)} Hz; the 3rd harmonic (blade-pass, ≈96–110 Hz) dominates.`,
  `**Final model:** two-stage logistic regression on 2 features (number of fan harmonics ≥ 6 dB; comb fundamental f0), trained on real + calibrated noise-augmented trials.`,
  `**Accuracy on real data:** ${cv.HLR_aug.loro_correct}/${cv.HLR_aug.loro_n} leave-one-round-out (95 % CI ${pct(cv.HLR_aug.ci95_lo)}–${pct(cv.HLR_aug.ci95_hi)}); repeated 5-fold × 10: ${pm(cv.HLR_aug.rep5x10_mean, cv.HLR_aug.rep5x10_sd)} %.`,
  `**Robustness:** 100 % under every calibrated realistic household condition for this phone; 17 noise types tested from realistic to destructive levels; failures are physical limits (supply-voltage speed drift > 3 %, a foreign tone on a fan line, much worse sensors).`,
  `**MATLAB port:** reproduces the Python model on all 80 recordings (max f0 difference ${D.matlab.df0} Hz, identical predictions ${D.matlab.same}).`,
]);

// =================================================================== TOC
body.push(new Paragraph({ children: [new PageBreak()] }));
body.push(new Paragraph({ children: [new TextRun({ text: "Contents", font: FONT, size: 32, bold: true, color: "1F3864" })], spacing: { after: 200 } }));
body.push(new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }));

// =================================================================== ABSTRACT
H1("Abstract");
P(`A smartphone taped to the wall next to a household wall fan was used as a vibration sensor to recognise four operating states of the fan: off, speed 1, speed 2 and speed 3. Eighty seven-second recordings (twenty per class) were made with the phone's ${p1.sensor.replace("tdk ", "TDK ").toUpperCase().replace("TDK ICM45621_ACC", "TDK ICM-45621")} MEMS accelerometer through the Phyphox app. The work follows the course brief in three phases — acquisition and audit, signal processing and features, classification and validation — and extends it with a calibrated study of robustness to noise.`);
P(`Phase 1 established that the sensor is clocked by a hardware oscillator at ${f(p1.fs_fit_mean, 3)} Hz with sub-microsecond timestamp scatter and no dropped samples, and that **linear interpolation onto a 400 Hz grid fakes a start-locked amplitude decay of up to ${f(100 * (1 - p1.linear_amp_min), 0)} %** of the dominant line; a Kaiser-windowed sinc interpolator removes it completely. Phase 2 showed that the running fan produces a harmonic comb k·f0 whose third harmonic dominates; f0 separates the three speeds by about 7 % with a within-class spread of only 0.5–0.7 %. An envelope-spectrum test indicates that f0 is the shaft rate (≈ 1925–2200 rpm) and the dominant line the blade-pass frequency of a three-blade rotor.`);
P(`Seventeen noise types — including Gaussian, Johnson–Nyquist (thermal), salt-and-pepper, speckle, flicker, Brownian and burst noise, footsteps, knocks, mains hum, a foreign machine, re-mounting, supply-voltage speed drift, dropped samples and lower sampling rates — were modelled and **calibrated against the recordings themselves, accelerometer datasheets and standards**. The measured noise floor (${f(ev.floor.nd_ug, 1)} µg/√Hz, Gaussian, kurtosis ${f(ev.floor.kurt)}) is reproduced by the synthetic model to within ${f(100 * Math.abs(ev.floor.synthetic_sigma_ratio - 1), 1)} %.`);
P(`Eight classifiers were compared under leakage-free, trial-level cross-validation with nested feature selection. The final model is a **two-stage logistic regression** ("is the fan on?" → "which speed?") on two physics-derived, gain- and orientation-invariant features, trained on the real trials plus calibrated noise-augmented copies with a worst-noise-group feature-selection objective. It classifies ${cv.HLR_aug.loro_correct}/${cv.HLR_aug.loro_n} trials in leave-one-round-out validation (95 % CI ${pct(cv.HLR_aug.ci95_lo)}–100 %; repeated 5-fold ${pm(cv.HLR_aug.rep5x10_mean, cv.HLR_aug.rep5x10_sd)} %) and stays at 100 % under all calibrated realistic household disturbances. Its limits are physical, not statistical: supply-voltage changes that shift the fan speed by more than ≈ 3 %, a foreign tone sitting exactly on a fan line, and sensors an order of magnitude noisier than this phone's. A MATLAB R2018a port of the final model was verified against Python on all recordings.`);

// =================================================================== 1 INTRODUCTION
H1("1. Introduction");
H2("1.1 Problem and goal");
P("Rotating machines leave a fingerprint in the vibration they transmit to their mounting: lines at the rotation frequency, at the blade-pass frequency and at their harmonics. The goal of this project is to read that fingerprint with the most widely available vibration sensor there is — the MEMS accelerometer inside a smartphone — and to classify four operating conditions of a wall fan: **off, speed 1, speed 2, speed 3** (regulator positions, oscillation disabled).");
H2("1.2 Why it is harder than it looks");
BUL([
  `**The vibration is tiny.** The fan adds only ${f(ev.fan.fan_line_rms * 1e3, 1)} mm/s² RMS in its dominant line; over the full band the fan's vibration power is only ${f(ev.fan.native_snr_db, 1)} dB above the sensor noise. Everything rests on narrow-band (spectral) processing.`,
  "**The phone is not a laboratory sensor.** Timestamps come from an operating system, the true sampling rate is not the nominal one, gravity is mixed into every axis and the phone's orientation on the wall is arbitrary.",
  "**The speed classes are close.** The three speeds differ by only ≈ 7 % in rotation rate, so frequency must be estimated to a fraction of a hertz.",
  "**A perfect score on clean data proves little.** All recordings come from one evening, so the classifier must be shown to survive the disturbances of other conditions — which is why the noise study was calibrated against real measurements instead of guessed.",
]);
H2("1.3 Contributions");
BUL([
  "A complete, reproducible pipeline (`py run_all.py`) from raw Phyphox exports to every table and figure in this report.",
  "Discovery and explanation (with a closed-form model) of a resampling artefact that linear interpolation produces when the device rate differs slightly from the analysis rate.",
  "A harmonic-comb feature extractor that is invariant to phone orientation and vibration gain and robust to impulses, mains hum and broadband noise.",
  "Seventeen calibrated noise models, each with an explicit evidence type (measured / datasheet / literature / assumption).",
  "A two-stage, noise-augmented logistic-regression classifier selected by a worst-group criterion, validated with confidence intervals and mean ± standard deviation over repeated cross-validation, and an ablation of every robust design choice.",
  "A Python command-line demo and a verified MATLAB port of the final model.",
]);
H2("1.4 How this report maps to the brief and rubric");
TABLE(["Rubric component (weight)", "Where it is addressed"], [
  ["Data quality & documentation (20 %)", "Section 2 (setup, audit, Week-1 note), Appendix A (reproducibility)"],
  ["Signal-processing correctness (30 %)", "Section 3 (timing, resampling, despiking, gravity removal, windowing, FFT resolution, aliasing)"],
  ["Feature discriminability / analysis depth (25 %)", "Sections 4–5 (features, FFT/CWT, rotation test, discriminability, noise calibration)"],
  ["Classification result & validation (15 %)", "Sections 6–7 (models, leakage-free CV, CIs, mean ± sd, robustness, ablation)"],
  ["Report / presentation clarity (10 %)", "This document; Section 8 (demo, MATLAB port)"],
], [3600, 6000], "Rubric map.", { left: true });

// =================================================================== 2 SETUP
H1("2. Experimental setup and data acquisition (Week 1)");
H2("2.1 Apparatus");
BUL([
  "**Source:** household wall fan, oscillation (sweep) switched off for every trial; the regulator positions define the classes.",
  `**Sensor:** ${p1.device}; accelerometer ${p1.sensor.replace("tdk icm45621_acc", "TDK ICM-45621")}; range ±${f(p1.range, 1)} m/s² (±8 g); reported resolution ${p1.reported_resolution} m/s²; minimum sampling interval ${f(p1.min_delay_us, 0)} µs (400 Hz).`,
  "**App:** Phyphox → Raw Sensors → *Acceleration (with g)*. Gravity was deliberately kept in the data; removing it is part of the processing.",
  "**Mounting:** phone taped flat to the wall beside the fan's mounting bracket, orientation fixed and its outline marked so that it can be re-mounted identically. The phone's axes are wall-relative: y vertical (gravity), z normal to the wall.",
]);
H2("2.2 Protocol");
BUL([
  "Four classes × 20 trials = 80 recordings; each a separate timed run with a 2 s start delay and 7 s duration, of which the central 5 s are analysed.",
  `**Round-robin order** (off → speed1 → speed2 → speed3, repeated 20 times; verified: ${p1.round_robin}). This spreads slow drift (temperature, mains voltage, tape creep) evenly over the classes so that it cannot masquerade as a class difference, and it defines natural validation folds (one "round" = one trial of each class recorded within ≈ 4 minutes).`,
  "After each regulator change the fan was allowed to settle before recording; the edge check below confirms no spin-up transient remains.",
]);
H2("2.3 Acquisition audit");
P("Every file was audited before any analysis (src/load_trials.py, src/phase1_audit.py). Timing is assessed by assigning each sample to an integer slot of the sensor clock and regressing time on slot index; the slope is the true rate, the residual is pure timestamp noise and missing slots are dropped samples.");
TABLE(["Quantity", "Result", "Interpretation"], [
  ["Trials per class", "20 / 20 / 20 / 20", "balanced"],
  ["Fitted sampling rate", `${f(p1.fs_fit_mean, 3)} Hz (range ${f(p1.fs_fit_min, 3)}–${f(p1.fs_fit_max, 3)})`, "hardware ODR; 0.027 % above nominal 400 Hz"],
  ["Timestamp scatter about the clock fit", `${f(p1.ts_resid_us_median, 2)} µs median (max ${f(p1.ts_resid_us_max, 2)} µs)`, "negligible vs 2500 µs period"],
  ["Dropped samples", `${p1.dropped_total}`, "no gaps in 80 × 7 s"],
  ["Duration", `${f(p1.duration_min, 2)}–${f(p1.duration_max, 2)} s`, "≥ 5 s analysis window + 1 s margins"],
  ["|g| (mean ± sd over trials)", `${f(p1.g_norm_mean, 4)} ± ${f(p1.g_norm_sd, 4)} m/s²`, "stable sensitivity"],
  ["Tilt in / out of wall plane", `${f(p1.tilt_inplane_mean, 2)}° / ${f(p1.tilt_outplane_mean, 2)}°`, "phone square on the wall"],
  ["Gravity-direction drift over session", `${f(p1.orientation_drift_deg, 2)}°`, "tape creep only"],
  ["Spacing of reported values", `${f(p1.value_spacing, 4)} m/s²`, "quantisation noise ≈ 0.1 × sensor noise"],
  ["RMS of first / last second vs centre", `${f(p1.edge_first_median, 3)} / ${f(p1.edge_last_median, 3)} (max ${f(p1.edge_max, 2)})`, "no spin-up or handling transient"],
  ["Flagged files", p1.flagged.length ? p1.flagged.join(", ") : "none", ""],
], [3000, 3300, 3300], "Acquisition audit (all 80 trials).");
FIGURE("p1_audit.png", "Audit over the 69-minute session: (a) fitted device sampling rate per trial, (b) timestamp scatter about the clock fit, (c) change of the mean (gravity) vector — the tape creeps by a few mm/s², i.e. < 0.1°.");
NOTE("Week-1 note (deliverable)", [
  `**Mounting method:** phone taped flat (screen out) to the wall beside the fan bracket, portrait, outline marked; oscillation off; gravity left in the data.`,
  `**Achieved sampling rate:** ${f(p1.fs_fit_mean, 3)} ± ${f((p1.fs_fit_max - p1.fs_fit_min) / 2, 3)} Hz (hardware clock fit), timestamp scatter ${f(p1.ts_resid_us_median, 1)} µs, zero dropped samples, 2790–2801 samples per 7 s recording.`,
  `**Dataset:** 80 tri-axial recordings, 20 per class, round-robin order, one session (2026-11-04, 22:02–23:12).`,
]);

// =================================================================== 3 SIGNAL PROCESSING
H1("3. Signal processing (Phase 1)");
P("The processing chain (src/preprocess.py) is, in order: **clock regularisation → Hampel despiking → band-limited resampling to exactly 400 Hz → zero-phase high-pass (gravity removal) on the full record → central 5 s window**. The order is deliberate and each step is justified below.");
H2("3.1 Timing model");
P("Android stamps each sample when the operating system receives it, but a FIFO-buffered MEMS sensor samples on its own oscillator. The correct model is therefore a perfectly uniform grid at an unknown rate. Each sample is assigned to the slot n = round((tₙ − t₀)/Δt_median), and the rate is estimated by least squares:");
EQ("tₙ ≈ t₀ + n / f_s,in ,        f_s,in = 1 / slope", "1");
P(`For all trials f_s,in = ${f(p1.fs_fit_min, 3)}–${f(p1.fs_fit_max, 3)} Hz and the timestamps scatter by only ${f(p1.ts_resid_us_median, 1)} µs about the fit, so treating the samples as uniform at f_s,in is exact for practical purposes. Missing slots (none in this data, but present in the dropout stress tests) are bridged and flagged so that later stages can ignore them.`);
H2("3.2 Resampling to a common 400 Hz grid — and an artefact discovered on the way");
P("All trials are mapped onto exactly 400.000 Hz so that every spectrum shares one frequency axis. The first implementation used linear interpolation, which is commonly considered adequate when timestamp jitter is small. It is not adequate here, and the reason is instructive.");
P("Linear interpolation at a fractional offset μ between two input samples is a two-tap filter with frequency response");
EQ("H(μ, ω) = 1 − μ + μ·e^(−jω),        |H|² = 1 − 2μ(1 − μ)(1 − cos ω),        ω = 2πf / f_s", "2");
P(`At f = 103.5 Hz (speed2's dominant line) ω = 1.63 rad and |H| falls from 1 at μ = 0 to cos(ω/2) = 0.65 at μ = ½. Because the input grid (400.108 Hz) and output grid (400 Hz) slide past each other, μ is not constant: μ(t) = frac(t · (f_s,in − 400 Hz)), which sweeps from 0 to ≈ 0.75 over a 7 s record. The interpolator therefore applies a **slowly varying, start-locked attenuation** to every high-frequency line. In the analysed window the dominant line was on average only ${f(p1.linear_amp_in_window, 2)} of its true amplitude, and at worst ${f(p1.linear_amp_min, 2)}.`);
P("This is exactly the \"start-locked decay\" first noticed during the data-quality check, which had been tentatively attributed to handling. Figure 2 shows that equation (2) predicts the measured amplitude profile essentially perfectly — the decay is a processing artefact, not physics. The cure is band-limited interpolation: a Kaiser-windowed sinc kernel");
EQ("h(d) = r · sinc(r·d) · I₀(β·√(1 − (d/L)²)) / I₀(β),     L = ⌈16/r⌉ input samples,  β = 8,  r = min(1, f_out/f_in)", "3");
P("whose pass-band is flat to ≈ 0.45 f_s (stop-band ≈ −80 dB). With it the line amplitude is constant to within 0.0 % over the record. The factor r lowers the cut-off when the output rate is lower than the input rate, so the same routine also performs anti-aliased decimation in the low-sampling-rate tests.");
FIGURE("p1_resampling.png", "Dominant-line amplitude over the recording after resampling 400.108 → 400 Hz. Red: linear interpolation (rejected); black dashed: closed-form prediction of equation (2); blue: Kaiser-windowed sinc (used). Grey: analysed window.", 0.9);
H2("3.3 Despiking");
P("A Hampel identifier replaces any sample further than 6 robust standard deviations from its local median (window ±5 samples = 27.5 ms):");
EQ("|xₙ − medₙ| > 6 · max(1.4826·MADₙ, σ_record)  ⇒  xₙ ← medₙ", "4");
P("It runs **before** resampling, because a sinc kernel would smear a single corrupt sample over 32 output samples. A record-wide robust σ is used as a floor so that coarsely quantised signals (local MAD = 0) are not turned into staircases. On the 670 128 real samples it changes nothing (expected Gaussian false-alarm rate ≈ 2·10⁻⁹), yet it is what makes the classifier immune to salt-and-pepper noise (Section 7.4).");
H2("3.4 Gravity removal");
P("Gravity contributes a constant ≈ 9.78 m/s² on the y axis, plus a tiny drift as the tape creeps. The brief allows mean subtraction or a high-pass filter; mean subtraction removes only the constant, while a high-pass also removes the slow drift and any building sway. The chosen filter is a **4th-order Butterworth high-pass at 2 Hz, applied forward and backward** (scipy `sosfiltfilt`, MATLAB `filtfilt`):");
BUL([
  "**Zero phase:** forward–backward filtering cancels the phase, so transients are not shifted; the magnitude becomes |H|² (effective 8th order, −6 dB at 2 Hz).",
  "**Cut-off:** far above DC and tape creep, and 16× below the lowest fan line (≈ 32 Hz), where the attenuation is 1 − (1 + (2/32)⁸)⁻¹ ≈ 2·10⁻¹⁰ — i.e. the fan band is untouched.",
  "**Order:** 4 gives a steep enough skirt for 2 Hz → 32 Hz while keeping the impulse response short (time constant ≈ 0.1 s).",
  "**Second-order sections** avoid the numerical problems of a high-order transfer function at a very low normalised cut-off (2/200 = 0.01).",
  "**Edge handling:** the filter runs on the full 7 s record and only then is the central 5 s kept, so start-up transients fall in the discarded 1 s margins.",
]);
FIGURE("p1_gravity_removal.png", "(a) Magnitude of the forward-backward high-pass; the fan band is shaded. (b) Low-frequency spectrum of the gravity (y) axis: raw, mean-subtracted and high-passed.");
H2("3.5 Windowing, spectral estimation, resolution and aliasing");
BUL([
  "**Window:** Hann everywhere (first sidelobe −31 dB, equivalent noise bandwidth 1.5 bins) — a good compromise between leakage from the strong dominant line and resolution.",
  "**FFT of the 5 s window:** N = 2000 → bin spacing f_s/N = 0.2 Hz, effective resolution ≈ 1.5 × 0.2 = 0.3 Hz; zero-padding to 4096 points only interpolates the plotted spectrum. Amplitudes are coherent-gain corrected (a sinusoid of amplitude A reads A).",
  "**Welch (mean) for plots:** 1024-point segments, 50 % overlap, 0.39 Hz bins.",
  "**Welch (median) for the robust features:** 512-point segments (0.78 Hz bins), 75 % overlap → 12 segments; the median ignores events present in fewer than half of them (footsteps, knocks, residual spikes). Its constant low bias for noise bins cancels in the SNR ratio. Sub-bin frequency accuracy is restored by parabolic interpolation of log-power around the peak.",
  "**Resolution is sufficient:** adjacent speed classes are 2.4 Hz apart in f0 (7.2 Hz at the dominant line), more than 9× the 0.78 Hz bin; the measured within-class f0 spread is 0.17–0.25 Hz.",
  "**Orientation invariance:** per-axis PSDs are summed, which equals the PSD of the acceleration vector and does not depend on how the phone is rotated.",
  "**Aliasing:** Nyquist is 200 Hz. The comb uses harmonics k ≤ 4 (≤ 147 Hz) and only those below 0.45 × the device rate. The 6th harmonic of speed3 (220 Hz) would alias to 180 Hz, but the measured noise floor falls by ≈ 25 dB between 100 and 200 Hz (Figure 11a), evidence of the sensor's on-chip anti-alias filter. The low-rate stress tests (Section 7) quantify what happens without it.",
]);

// =================================================================== 4 FEATURES
H1("4. Feature extraction and analysis (Phase 2)");
P("Twenty-nine features are computed per trial (src/features.py), grouped by what they measure. The guiding question for each one is not only *does it separate the classes today?* but *will it still mean the same thing on another day, mount or phone?*");
H2("4.1 Time-domain features");
P("RMS, peak-to-peak, crest factor (peak/RMS) and kurtosis are computed on x, y, z and on the **principal vibration axis** pc1 (projection on the eigenvector of the 3×3 covariance with the largest eigenvalue — an orientation-invariant scalar), plus the vector RMS. Intuition: RMS measures vibration energy (fan on vs off); a pure tone has kurtosis 1.5 and crest √2, Gaussian noise 3 and ≈ 3–4, so kurtosis and crest drop when a tone dominates. Their weakness is equally clear: every amplitude feature changes with how firmly the phone is taped, with the sensor's noise level and (for single axes) with orientation.");
H2("4.2 Frequency-domain features");
P("FFT dominant frequency and amplitude (z axis — the textbook choice — and vector spectrum), Welch dominant frequency, spectral centroid (20–190 Hz), spectral flatness (geometric/arithmetic mean of the PSD: 1 for white noise, → 0 for lines) and the share of energy in 90–115 Hz.");
H2("4.3 Continuous wavelet transform");
P(`The CWT uses a complex Morlet wavelet implemented in the frequency domain: at centre frequency f_c it is a Gaussian band-pass with σ_f = f_c / ${D.config.CWT_CYCLES} (constant Q), i.e. 4.2 Hz and σ_t = 38 ms at 100 Hz, amplitude-normalised so that a sinusoid of amplitude A gives |W| = A. The scalograms (Figure 7) confirm that the fan lines are stationary over the window — there is no spin-up, beating or intermittency — which justifies spectral (time-averaged) features. The CWT dominant frequency agrees with the FFT within ${f(p2.cwt_vs_fft_max_diff_hz, 2)} Hz on every trial.`);
H2("4.4 The harmonic-comb features (the key engineered features)");
P("The running fan does not produce one line but a comb at k·f0 (k = 1…5, Figure 5). Using the whole comb is far more robust than picking the largest peak. On the median-Welch vector PSD P(f), a local noise floor F(f) is estimated by a running median over ±10 Hz, giving SNR(f) = 10·log₁₀(P/F). For each candidate f0 on a 0.02 Hz grid:");
EQ("S(f0) = (1/K′) · Σₖ min( max( SNR(k·f0), 0 ), 30 dB ),     k = 1…4,  k·f0 < 0.45·f_s,in,  k·f0 ∉ 50/100/150 ± 1 Hz", "5");
P("The best candidate gives **comb_snr** = max S (detection statistic) and **f0** (refined as the SNR-weighted mean of the parabolic peak positions divided by k); **n_harm** counts its harmonics ≥ 6 dB and **line_snr** is the strongest one. Every design choice has a reason, and each was verified by the ablation in Section 7.4:");
BUL([
  "**Local-floor SNR** instead of absolute power → invariant to vibration gain (looser tape) and to the phone's noise level.",
  "**Cap at 30 dB per harmonic** → one strong foreign line cannot dominate the score.",
  "**Mains mask (±1 Hz around 50/100/150 Hz)** → magnetostriction hum is fixed-frequency and must never count as a fan harmonic. The calibration (Section 5) found such a 100 Hz hum in the data — produced by the fan motor itself — at 6–10 dB.",
  "**f0 restricted to 28–42 Hz** (observed 31.7–37.2 Hz plus ±10 % headroom) → excludes the k = 2 and k = 4 aliases of the dominant line (e.g. 103.6/4 = 25.9 Hz), which otherwise cause harmonic-number errors when noise buries all but one harmonic. This was found by the noise study, not assumed.",
  "**Harmonics only below 0.45 × the device rate** → the comb adapts automatically to phones that sample slower.",
]);
H2("4.5 Results: frequencies and the rotation question");
TABLE(["Class", "FFT dominant (Hz)", "CWT dominant (Hz)", "f0 (Hz)", "rpm if shaft = f0", "harmonics ≥ 6 dB"],
  D.freq.map((r) => [r.cls, `${f(r.fft_fdom_mean)} ± ${f(r.fft_fdom_sd)}`, f(r.cwt_fdom_mean), `${f(r.f0_mean)} ± ${f(r.f0_sd)}`, f(r.rpm_if_rotation_is_f0, 0), r.harmonics_present]),
  [1200, 1900, 1700, 1700, 1600, 1500], "Dominant frequency per class (mean ± sd over 20 trials). The dominant line is 3·f0.");
P("The brief asks to compare the dominant frequency with the expected rotational speed. The initial expectation (15–25 Hz rotation, blade-pass = rotation × blades) does not fit directly: the dominant line is at 96–110 Hz and the lowest comb member at 32–37 Hz. Two hypotheses remain: (A) the shaft turns at f0 (≈ 1925–2200 rpm, a 2-pole motor with high slip) and the dominant 3·f0 line is the blade-pass frequency of a 3-blade rotor; (B) the shaft turns at f0/3 (≈ 640–735 rpm) and f0 itself is the blade-pass. Under (B) an unbalanced rotor would produce a once-per-revolution line at f0/3 and modulate the blade-pass line at f0/3. Figure 8 tests both: the envelope spectrum of the dominant band shows no peak at f0/3 (prominence " + Object.values(p2.envelope).join("; ") + "), and the spectrum has no line at f0/3 or 2f0/3, while a clear 1× line exists at f0 for speeds 2 and 3. **The evidence supports hypothesis A: shaft at f0 ≈ 1925–2200 rpm, dominant line = blade-pass of a 3-blade rotor.** Counting the blades and a slow-motion video would confirm it.");
FIGURE("p2_time_series.png", "Representative trial of each class: gravity-free acceleration in the 5 s window (x/y/z offset) and the first 100 ms.");
FIGURE("p2_fft_per_class.png", "Amplitude spectra (Hann, 5 s, vector of the three axes) of the representative trials with the comb k·f0 marked; grey bands are the masked mains frequencies.");
FIGURE("p2_psd_all.png", "Welch PSD of all 80 trials coloured by class, with a zoom on the dominant line.");
FIGURE("p2_cwt_per_class.png", "Morlet CWT scalograms of the principal axis; the dotted line marks 3·f0. The lines are stationary over the window.");
FIGURE("p2_envelope_rotation_test.png", "Rotation test: (a) envelope spectrum of the 3f0 ± 15 Hz band, (b) low-frequency spectrum. Dotted lines mark f0/3; no modulation or line appears there.");
H2("4.6 Feature discriminability");
TABLE(["Feature", "off", "speed1", "speed2", "speed3"],
  D.per_class.map((r) => [r.feature, ...["off", "speed1", "speed2", "speed3"].map((c) => {
    const [m, s] = r[c]; const d = Math.abs(m) < 0.1 ? 4 : Math.abs(m) < 10 ? 3 : 2; return `${f(m, d)} ± ${f(s, d)}`; })]),
  [1800, 1950, 1950, 1950, 1950], "Key features per class (mean ± sd over 20 trials; RMS in m/s², frequencies in Hz, SNRs in dB).");
TABLE(["Feature", "Fisher ratio", "worst class-pair gap (sd)", "fully separated pairs (of 6)"],
  D.ranking.slice(0, 12).map((r) => [r.feature, f(r.fisher, 1), f(r.worst_pair_gap_sd, 2), r.separable_pairs]),
  [2800, 2200, 2400, 2200], "Top features ranked by Fisher ratio (between-class / within-class variance) on clean data.");
P("Two lessons come out of the ranking. First, **amplitude features (RMS) have the largest Fisher ratios on clean data** because they separate off from on enormously — but Section 7 shows they collapse under noise or a looser mount. Discriminability on clean data is not robustness. Second, **no single feature separates all four classes**: frequency features separate the three speeds (the FFT dominant frequency separates every pair, worst gap " + f(D.ranking.find((r) => r.feature === "fft_fdom_vec").worst_pair_gap_sd, 1) + " sd) but are meaningless for *off*, where they lock onto noise; energy/SNR features separate off from on but barely order the speeds. This structure — *detect, then estimate* — is what the final two-stage model encodes.");
FIGURE("p2_feature_ranking.png", "Fisher ratio of every feature on clean data (green: all six class pairs fully separated).", 0.9);
FIGURE("p2_feature_scatter.png", "Feature pairs: the textbook pair (z-RMS, FFT peak), the physics pair (comb SNR, f0) and shape statistics.");

// =================================================================== 5 NOISE
H1("5. Noise modelling and calibration");
H2("5.1 Why calibrate, and how");
P("All 80 recordings come from a single quiet evening, so a perfect score says nothing about other conditions, and more recordings were not possible. The robustness of the classifier is therefore studied by injecting synthetic noise into the real recordings. Synthetic noise is only meaningful if its **type and level are tied to reality**, otherwise a robustness claim can be made arbitrarily good or bad by choosing the level. Every noise model was therefore calibrated (src/calibrate_noise.py) and labelled with its evidence type:");
BUL([
  "**measured** — from the recordings themselves: fan-off trials are the real noise floor; fan-on trials give real line amplitudes, frequency stability and hum; the audit gives timing and orientation.",
  "**datasheet** — accelerometer noise densities and ADC arithmetic for other phones.",
  "**literature** — standards and published measurements (supply voltage, floor vibration, Android sampling limits).",
  "**assumption** — plausible but unmeasured (flagged); **uncalibrated** — amplitude unknowable without new recordings, reported as a stress test only.",
]);
P("Each family is swept from its realistic range into destructive stress levels; results always distinguish the two. Physically generated noises (thermal, flicker, burst, impacts) are passed through the measured spectral shape of the sensor (a 129-tap linear-phase FIR fitted to the fan-off PSD), because everything the phone reports has passed through its anti-alias filter.");
H2("5.2 The real noise floor");
P(`The fan-off trials give a plateau noise density of **${f(ev.floor.nd_ug, 1)} µg/√Hz** per axis (σ = ${f(ev.floor.sigma * 1e3, 2)} mm/s² after high-pass), consistent with the 70 µg/√Hz datasheet value of the same TDK sensor family. The samples are Gaussian (kurtosis ${f(ev.floor.kurt)} — Gaussian is 3 — range ${f(ev.floor.kurt_range[0])}–${f(ev.floor.kurt_range[1])} over trials and axes; skewness ${f(ev.floor.skew, 3)}) and the axes are independent (mean |r| = ${f(ev.floor.mean_abs_axis_corr, 3)}). The flicker (1/f) corner is at ${f(ev.pink.fc, 2)} Hz, far below the 2 Hz high-pass. **Validation:** synthetic noise generated from the fitted model and passed through the same pipeline has σ within ${f(100 * Math.abs(ev.floor.synthetic_sigma_ratio - 1), 1)} % of the real floor and a spectrum within ${f(ev.floor.synthetic_psd_rms_db_err, 1)} dB RMS (Figure 11d).`);
FIGURE("calib_noise_floor.png", "Noise-floor calibration: (a) fan-off PSD per axis vs datasheet densities (dotted, µg/√Hz); (b) amplitude distribution vs Gaussian; (c) low-frequency floor and 1/f fit; (d) synthetic noise model vs real, through the same pipeline.");
H2("5.3 The fan signal and its natural variability");
P(`Fan vibration power (3 axes, broadband) is only ${f(ev.fan.native_snr_db, 1)} dB above the floor; the dominant line has ${f(ev.fan.fan_line_rms * 1e3, 2)} mm/s² RMS. Within the session f0 varied by ${Object.entries(ev.fan.f0_sigma_pct).map(([c, v]) => `${c} ${f(v, 2)} %`).join(", ")} (sd) with no systematic drift (< 0.4 %/h), and the line amplitude by ${Object.values(ev.fan.line_cv_by_class).map((v) => f(100 * v, 1)).join("/")} % between trials. The amplitude "wobble" within a trial (envelope CV ${f(ev.fan.cv_raw, 3)}) is fully explained by additive noise; its raw value is used as an upper bound for speckle. A weak 100 Hz line appears only when the fan runs (largest ${f(ev.mains_hum.max_line_snr_db, 1)} dB above the floor): stator magnetostriction of the fan's own motor at twice the mains frequency (Figure 12c).`);
FIGURE("calib_fan_and_hum.png", "(a) Same-day fan-speed stability, (b) amplitude wobble of the dominant line, raw vs after removing the additive-noise contribution, (c) mains-hum line SNR at the mount by class.");
H2("5.4 The noise families");
P("Every family takes the raw recording (timestamps and 3-axis acceleration with gravity) and returns what the phone would have reported, so the complete pipeline is exercised (src/noise_models.py). The physics and the calibration of each:");
const famText = [
  ["Gaussian (AWGN)", "Textbook additive white Gaussian noise, i.i.d. per axis, specified as SNR relative to the fan's broadband vibration power. Real phone noise is Gaussian (measured kurtosis 3.00), so this models a noisier phone. Realistic: this phone adds nothing; the noisiest phone part (MPU-6050 class, 400 µg/√Hz) corresponds to " + f(D.cal_bands.gaussian[0], 1) + " dB."],
  ["Johnson–Nyquist (thermal)", "Thermal noise of the MEMS proof mass, the mechanical form of the fluctuation–dissipation theorem (Gabrielson 1993): S_a = 4·k_B·T·ω₀ / (m·Q) — white, Gaussian and proportional to absolute temperature. Worst case assumed: the whole measured floor is thermal. Heating from 25 °C to 45 °C raises the noise power by only 6.7 % (+0.3 dB). A second sweep scales 1/(mQ) — a smaller proof mass — which maps onto cheaper sensors (×1.44 for 70 µg/√Hz parts, ×2.94 for 100, ×47 for 400)."],
  ["Salt-and-pepper", "A fraction of samples per axis replaced by +full-scale (salt) or −full-scale (pepper), ±78.45 m/s² — corrupt words on the sensor bus or saturation glitches. Measured: 0 outliers in 670 128 samples, so the real rate is < 4.5·10⁻⁶ (rule of three, 95 %)."],
  ["Speckle", "Multiplicative noise, defined as in MATLAB imnoise: J = I + n·I with n uniform, zero mean, variance v, applied to the vibration (not to gravity) — gain fluctuation. Realistic upper bound from the measured line-amplitude wobble: v ≤ " + f(D.cal_bands.speckle[1], 2) + "."],
  ["Pink (flicker, 1/f)", "Electronic flicker noise with PSD equal to the white floor at a corner f_c; measured f_c = " + f(ev.pink.fc, 2) + " Hz."],
  ["Brownian (1/f², random walk)", "Bias random walk. Measured upper bound from the gravity-vector drift between fan-off trials (includes tape creep): " + sci(ev.brownian.sigma_rw) + " m/s²/√s."],
  ["Popcorn (burst / random telegraph)", "Offset jumping between two levels at Poisson times (5 switches/s). Measured: no step larger than " + f(ev.popcorn.max_step_over_sigma, 1) + " σ in 50 ms block means; detectability bound " + sci(ev.popcorn.bound) + " m/s²."],
  ["Quantisation", "Rounding to a coarser ADC step (gravity included, as in a real sensor). This phone: " + f(ev.adc.value_spacing, 4) + " m/s² spacing. Other phones: 16-bit ±2 g … ±16 g → 0.0006–0.0048 m/s²; 12-bit ±8 g → 0.038 m/s²."],
  ["Footsteps", "Heel strikes at 1.6–2.2 steps/s exciting floor/wall modes of 8–90 Hz (3–8 % damping). Floors are designed to keep walking peaks below 0.5 % g ≈ 0.05 m/s² (AISC Design Guide 11); the largest event in 70 min of fan-off windows was " + f(ev.transients.max_abs_off_window, 3) + " m/s²."],
  ["Knocks / door slams", "1–3 impacts: 3 ms contact pulse + 4–6 lightly damped modes (10–180 Hz). Anchor: 0.41 m/s² measured directly on an impacted wooden floor (Ding et al. 2020); a wall sensor sees less, 0.1 m/s² assumed realistic."],
  ["Mains hum", "Fixed 100 Hz (+50/150 Hz) tone from transformers, ballasts or the motor stator. Measured at the mount: < " + sci(ev.mains_hum.amp_bound) + " m/s², i.e. ≤ " + f(D.cal_bands.mains_hum[1], 2) + " × the fan line."],
  ["Foreign machine (appliance)", "A 2-pole compressor (2850–2970 rpm → 47.5–49.5 Hz) with harmonics; its 2nd harmonic lands on speed1's dominant line — the adversarial case. Amplitude uncalibrated (needs a recording with the appliance running)."],
  ["Orientation (re-taping)", "Rigid rotation of the 3-axis signal including gravity. Measured in-session drift 0.09°; re-taping inside the marked outline assumed < 5°."],
  ["Coupling (mount gain)", "Structural vibration reaches the sensor with gain g < 1; the sensor's own noise is topped back up. Measured same-mount trial-to-trial variation ≈ 2 % → g ≥ " + f(D.cal_bands.coupling[0], 3) + "."],
  ["Speed shift (supply voltage)", "Band-limited time-scaling of all fan lines. Same day: measured f0 scatter → ±" + f(D.cal_bands.speed_shift[1], 2) + " %. Other days: the supply may vary by ±6 % (Indian Electricity Rules / state supply codes; ESMI measurements show frequent sags). From motor torque K·V²·s = fan load c·n² with n = n_s(1 − s): dn/n = (2s/(1 + s))·dV/V; the measured slips give 2s/(1+s) = " + f(ev.speed.kappa_lo, 2) + "–" + f(ev.speed.kappa_hi, 2) + ", so ±6 % voltage → ±" + f(ev.speed.day_pct_lo, 1) + "–" + f(ev.speed.day_pct_hi, 1) + " % speed."],
  ["Dropped samples", "Gaps of 20–150 ms (OS scheduling). This phone: 0 dropped; other devices show sampling instability (Stisen et al. 2015)."],
  ["Lower sampling rate", "Android 12+ caps motion sensors at 200 Hz unless an app holds HIGH_SAMPLING_RATE_SENSORS. Modelled with the sensor's on-chip anti-alias filter (cut-off 0.45 f_s) and, as a worst case, without it (aliasing)."],
  ["Calibrated mixes", "same day (household activity, this phone and mount), other day (+ re-taping and supply-voltage speed shift), other phone (+ a configuration drawn from the datasheet catalogue: noise density, ADC step, 400 or 200 Hz). Severity 1 = calibrated realistic, swept 0–4."],
];
TABLE(["Family", "Physical origin, model and calibration"], famText, [2300, 7300], "The noise families.", { left: true, size: 16 });
H2("5.5 Calibration summary");
TABLE(["Family", "Realistic range", "Evidence", "Basis"], D.cal_table.map((r) => [r.family, r.realistic, r.evidence_type, r.evidence]),
  [1500, 1900, 1500, 4700], "Calibrated realistic ranges (level units as in the noise-family table).", { left: true, size: 14 });

// =================================================================== 6 CLASSIFICATION
H1("6. Classification and validation (Phase 3)");
H2("6.1 The models compared");
TABLE(["Name", "What it is"], D.models.concat(["HLR_aug_known"]).map((m) => [m, MODEL_DESC[m]]), [2000, 7600], "Model names used throughout.", { left: true });
P("**T1_textbook** is what a first attempt would do and serves as the baseline. **T2_physics** uses the same two-stage threshold logic on the noise-hardened comb features; its off/on threshold is a *constant-false-alarm-rate* (CFAR) threshold set from the off class alone (mean + 4 sd, at least max + 0.5 dB): the comb SNR of an off trial is noise measured against noise, so its distribution does not move when noise rises, whereas a midpoint threshold would sit needlessly close to the on class. The ML models choose their own 2–3 features (Section 6.4).");
P("**Why a two-stage logistic regression.** For an off trial, f0 is meaningless (the comb locks onto noise anywhere in 28–42 Hz). A single multinomial logistic regression must express every class score as a linear function of all features, so it cannot represent \"f0 matters only when the fan is on\"; the off class, spread over all f0, then leaks into the speed classes. The physics of the problem is *detection followed by estimation*, and the model is built the same way:");
EQ("P(on) = σ(w_A·z_A + b_A),     P(speed k | on) = softmax_k(W_B·z_B + b_B)", "6");
EQ("P(off) = 1 − P(on),     P(speed k) = P(on)·P(speed k | on);     decide off if P(on) < 0.5", "7");
P("where z are standardised features. Stage A is trained on all rows (off vs on), stage B only on fan-on rows. Both stages are plain logistic regression, so the model stays within the brief (\"basic ML on 2–3 features\") while matching the structure of the problem.");
H2("6.2 Validation protocol and leakage control");
BUL([
  "**Every split is by trial.** A trial is one sample; windows of one trial never appear in both training and test.",
  "**Leave-one-round-out (LORO):** 20 folds, each holding out one complete round-robin cycle (one trial per class, recorded within ≈ 4 min) — no temporal neighbour of a test trial is ever trained on. LORO is deterministic, so it is reported with a Clopper–Pearson 95 % confidence interval and as the mean ± sd of the per-round accuracy.",
  "**Repeated stratified 5-fold CV (10 repeats):** reported as mean ± sd over the repeats.",
  "**Everything is fitted inside the training folds:** thresholds, standardisation, feature selection (nested — inner folds also split by trial) and model weights.",
  "**Why confidence intervals matter:** with 80 trials, 80/80 correct still only guarantees ≥ 95.5 % accuracy at 95 % confidence; with 20 test trials it would be ≥ 83 %.",
]);
H2("6.3 Noise-augmented training");
P("The *_aug models are trained on the real trials plus K = 4 corrupted copies of every training trial. Each copy applies 1–3 randomly chosen noise families at random levels from their augmentation ranges, in physical order (operating point → mounting → mechanical → electronic → acquisition). The copies of a trial are used only while that trial is in the training folds. Intuition: the model is shown, during training, what the same physical state looks like under disturbances, so it learns which features keep their meaning.");
H2("6.4 Feature selection: invariant pool and worst-group objective");
P("Selection is greedy forward (≤ 3 features), pre-filtered to the 12 highest Fisher ratios, scored by inner cross-validation split by trial. Two refinements were adopted after testing them under the strictest protocol — leave-one-noise-group-out, where the tested noise type never appears in training:");
BUL([
  "**Invariant candidate pool.** Only features that are invariant *by construction* to vibration gain and phone orientation are allowed: comb_snr, f0, n_harm, line_snr, spectral flatness / centroid, band ratio, Welch and vector-FFT dominant frequency, kurtosis and crest of pc1. Raw amplitudes and single-axis features are excluded — they encode the mount, not the fan.",
  "**Worst-group objective.** Each training row carries a mask of the noise groups used to corrupt it. Instead of maximising mean accuracy, selection maximises the *minimum* accuracy over the clean rows and each noise group: a feature that works only under some noise types is rejected. Intuition: a feature that survives every disturbance we can simulate is more likely to survive one we cannot.",
]);
TABLE(["Test (unseen noise type)", ...CFG.map((c) => c.replace(" (adopted)", " ✔"))],
  [...new Set(D.selection.map((r) => r.test))].map((t) => [t, ...CFG.map((c) => { const r = D.selection.find((x) => x.model === "kNN_aug" && x.test === t && x.config === c); return f(100 * r.acc, 1); })])
    .concat([["**mean**", ...CFG.map((c) => `**${f(100 * selMean("kNN_aug", c), 1)}**`)]]),
  [2400, 1800, 1800, 1800, 1800], "Selection design study, kNN_aug: accuracy (%) on noise types never seen in training (mean of 2 repeated 5-fold CVs).");
P(`For kNN the adopted procedure raises the mean accuracy on unseen noise from ${f(100 * selMean("kNN_aug", CFG[0]), 1)} % to ${f(100 * selMean("kNN_aug", CFG[3]), 1)} % (footsteps 49 → 99 %, knocks 78 → 97 %, 200 Hz sampling 27 → 81 %); for the two-stage logistic regression from ${f(100 * selMean("HLR_aug", CFG[0]), 1)} % to ${f(100 * selMean("HLR_aug", CFG[3]), 1)} %. Without these refinements, selection grabbed fragile features (line_snr, rms_z) whenever the tested noise type was absent from training.`);
H2("6.5 Results on the real recordings");
TABLE(["Model", "LORO correct", "95 % CI", "LORO per-round mean ± sd (%)", "5-fold × 10 mean ± sd (%)"],
  D.clean_cv.map((r) => [r.model, `${r.loro_correct}/${r.loro_n}`, `${pct(r.ci95_lo)}–${pct(r.ci95_hi)}`, pm(r.loro_round_mean, r.loro_round_sd), pm(r.rep5x10_mean, r.rep5x10_sd)]),
  [1800, 1500, 2000, 2100, 2200], "Accuracy on the 80 real recordings.");
FIGURE("p3_confusion_clean.png", "Pooled LORO confusion matrices of all models.");
const selRows = []; ["kNN_clean", "LR_clean", "HLR_clean", "kNN_aug", "LR_aug", "HLR_aug"].forEach((m) => D.selected.filter((r) => r.model === m).slice(0, 3).forEach((r) => selRows.push([m, r.features, `${r.folds}/${r.of}`])));
TABLE(["Model", "Selected features", "LORO folds"], selRows, [2000, 5600, 2000], "Features chosen inside the LORO folds (most frequent sets).");
P("The contrast is the point of the study: trained on clean data, selection picks amplitude and raw-peak features (rms_vec, fft_fdom_z); trained on noise-augmented data it picks the physics features (n_harm, f0, comb_snr) — on its own, without being told. The final model chose **n_harm + f0** in " + D.selected.find((r) => r.model === "HLR_aug").folds + "/20 folds.");
const mg = p3.margins;
P(`**Margins.** T2's closest off trial is ${f(mg.T2_min_snr_margin_off_db, 2)} dB below its detection threshold and its closest on trial ${f(mg.T2_min_snr_margin_on_db, 1)} dB above; the closest speed is ${f(mg.T2_min_f0_margin_hz, 2)} Hz from a speed boundary. The final model's probability for the true class, out of sample (LORO), is ${pm(D.demo.loro_ptrue_mean, D.demo.loro_ptrue_sd)} % (minimum ${pct(D.demo.loro_ptrue_min)}).`);
H3("The final model");
TABLE(["Stage", "Feature", "Standardisation (mean, scale)", "Coefficient(s)", "Intercept(s)"], [
  ["A: fan on?", A.features.join(", "), `${f(A.scaler_mean[0], 3)}, ${f(A.scaler_scale[0], 3)}`, f(A.coef[0][0], 3), f(A.intercept[0], 3)],
  ["B: which speed?", B.features.join(", "), `${f(B.scaler_mean[0], 3)}, ${f(B.scaler_scale[0], 3)}`, B.coef.map((c) => f(c[0], 3)).join(" / "), B.intercept.map((c) => f(c, 3)).join(" / ")],
], [1700, 1300, 2400, 2200, 2000], "Final model (trained on all 80 trials + augmentation; results/model/final_model.json).");
P(`In words: stage A declares the fan **on if at least ${Math.ceil(nStar)} harmonic of a consistent comb stands ≥ 6 dB above the local noise floor** (logit zero at n_harm = ${f(nStar, 2)}); stage B assigns **speed1 below f0 = ${f(b12, 2)} Hz, speed2 up to ${f(b23, 2)} Hz and speed3 above** — almost exactly the midpoints between the class means (${f((freq.speed1.f0_mean + freq.speed2.f0_mean) / 2, 2)} and ${f((freq.speed2.f0_mean + freq.speed3.f0_mean) / 2, 2)} Hz) and close to T2's thresholds (${f(p3.t2.f0_speed1_2_hz, 2)} / ${f(p3.t2.f0_speed2_3_hz, 2)} Hz). The learned model is physically interpretable.`);
FIGURE("p3_final_model_regions.png", "Decision regions of the final model (x jittered for display): large dots are real trials, small dots noise-augmented copies.", 0.75);
H3("Analysis-window length");
const wins = [5, 2, 1, 0.5];
TABLE(["Window", "T2_physics", "HLR_clean", "LR_clean", "f0 sd, speed2 (Hz)"],
  wins.map((w) => { const g = (m) => D.window.find((r) => r.window_s === w && r.model === m); return [`${w} s`, pm(g("T2_physics").rep_mean, g("T2_physics").rep_sd), pm(g("HLR_clean").rep_mean, g("HLR_clean").rep_sd), pm(g("LR_clean").rep_mean, g("LR_clean").rep_sd), f(g("T2_physics").f0_sd_speed2, 3)]; }),
  [1500, 2000, 2000, 2000, 2100], "Accuracy (%, repeated 5-fold × 10, mean ± sd) vs analysed window length.");
P("Shorter windows coarsen the spectrum (Δf ≈ 1/T) and fewer segments enter the median. The two-stage architecture stays at 100 % down to 0.5 s, whereas the threshold rule degrades below 1 s — a classification could be updated twice per second in a live demo.");
FIGURE("p3_window_length.png", "Accuracy vs analysed window length (mean ± sd over 10 repeated 5-fold CVs).", 0.7);

// =================================================================== 7 ROBUSTNESS
H1("7. Robustness to noise");
H2("7.1 Protocol");
BUL([
  "Repeated stratified 5-fold CV by trial (R = 5 repeats); models trained on the training folds only.",
  "Every **test** trial is corrupted with a fresh random draw of a noise family at each level and pushed through the complete pipeline; results are the **mean ± sd over the 5 repeats**.",
  "**Unseen noise (strict):** for the *_aug models, every family of the tested family's group is removed from the training augmentation (leave-one-group-out). **Known noise (deployment):** HLR_aug_known is the final model with all noise groups in its augmentation.",
  "Levels inside the calibrated realistic range are distinguished from stress levels in every result.",
]);
H2("7.2 Results");
const fams = Object.keys(D.families);
const cellR = (r) => (r && r.worst_realistic !== null ? `${f(100 * r.worst_realistic, 0)} ± ${f(100 * r.worst_realistic_sd, 0)}` : "n/a");
TABLE(["Noise family", "T1 textbook", "T2 physics", "kNN_aug (unseen)", "HLR_aug (unseen)", "HLR_aug_known"],
  fams.map((fm) => [fm, ...["T1_textbook", "T2_physics", "kNN_aug", "HLR_aug", "HLR_aug_known"].map((m) => cellR(RS(fm, m)))]),
  [2100, 1500, 1500, 1500, 1500, 1500], "Worst accuracy (%, mean ± sd over 5 repeats) within the calibrated realistic range. n/a = uncalibrated family.", { size: 16 });
TABLE(["Noise family", "T1 textbook", "T2 physics", "kNN_aug (unseen)", "HLR_aug (unseen)", "HLR_aug_known"],
  fams.map((fm) => [fm, ...["T1_textbook", "T2_physics", "kNN_aug", "HLR_aug", "HLR_aug_known"].map((m) => RS(fm, m).first_level_below_90)]),
  [2100, 1500, 1500, 1500, 1500, 1500], "First tested level at which mean accuracy falls below 90 % (\"none\" = never, up to the harshest level). Units as in the noise-family table.", { size: 16 });
FIGURE("p3_robustness_curves.png", "Accuracy vs noise level for every family and model (mean ± 1 sd shaded). Green: calibrated realistic range; gold: other-day supply-voltage range.");
FIGURE("p3_robustness_heatmap.png", "Summary: worst accuracy inside the realistic range (left) and at the harshest stress level (right), mean ± sd.", 0.95);
H2("7.3 Interpretation");
BUL([
  `**Household disturbances for this phone are harmless to the final model:** footsteps, knocks, hum, pink/Brownian/burst noise, salt-and-pepper, speckle, re-taping, same-mount coupling changes, dropped samples and the calibrated same-day mix all stay at 100 % within their realistic ranges. The textbook rule already fails at footsteps of ${t1("footsteps").first_level_below_90} m/s², hum of ${t1("mains_hum").first_level_below_90}× the fan line and a 90° rotation.`,
  `**Thermal (Johnson–Nyquist) noise is irrelevant at any realistic temperature** — its power scales with absolute temperature, so 15–45 °C changes the floor by < 0.5 dB; only a 10–30× smaller proof mass (a much worse sensor) matters.`,
  `**Gaussian noise:** the final model first drops below 90 % at an SNR of ${hu("gaussian").first_level_below_90} dB when Gaussian-type noise was never seen in training, and only at ${hk("gaussian").first_level_below_90} dB when it was. Modern phone accelerometers (70–100 µg/√Hz) correspond to ≥ ${f(ds["100"].awgn_snr_db, 1)} dB; only the oldest part (400 µg/√Hz, ${f(ds["400"].awgn_snr_db, 1)} dB) defeats every classifier — physically the fan line is then below the noise in every bin.`,
  `**Speed shift:** every classifier is at ≈ 100 % up to ±2 % and fails at ±3 % (first level below 90 %: ${hk("speed_shift").first_level_below_90} %). This is not a weakness of the classifier but of the measured quantity: the speed classes are only ≈ 7 % apart, so a supply-voltage change that moves the speed by half the class spacing makes the classes physically indistinguishable. The calibrated other-day mix therefore scores ${pm(hk("mix_other_day").worst_realistic, hk("mix_other_day").worst_realistic_sd, 0)} %. Remedy: one reference recording per speed at deployment, or measuring the supply voltage.`,
  `**Foreign tone on a fan line (appliance):** when the other harmonics are buried, a compressor line at 96 Hz *is* a speed1 signature; above 0.3× the fan line, off trials are called speed1/speed3. Requiring two harmonics fixes this but costs sensitivity (this fan's energy is concentrated in one line) — a genuine trade-off, not a bug.`,
  `**Other phones:** coarse ADCs (12-bit, 0.038 m/s²) and apps capped at 200 Hz remove the dominant line (it lies above 0.45 × 200 Hz); the calibrated other-phone mix scores ${pm(hk("mix_other_phone").worst_realistic, hk("mix_other_phone").worst_realistic_sd, 0)} %. Here the CFAR threshold of T2 is more sensitive than the learned detector (T2: ${pm(t2("lower_fs").worst_realistic, t2("lower_fs").worst_realistic_sd, 0)} % at 200 Hz).`,
  `**Unseen vs known noise:** the gap between HLR_aug and HLR_aug_known shows how much a known disturbance can be trained away (e.g. 400 µg/√Hz sensor noise: ${pm(hu("jn_proofmass").worst_realistic, hu("jn_proofmass").worst_realistic_sd, 0)} → ${pm(hk("jn_proofmass").worst_realistic, hk("jn_proofmass").worst_realistic_sd, 0)} %).`,
]);
H2("7.4 Ablation: what each robust design choice buys");
const ab = D.ablation;
const comps = [...new Set(ab.map((r) => r.component))];
TABLE(["Component removed", "Test condition", "T2 full", "T2 without", "HLR_clean full", "HLR_clean without"],
  comps.map((c) => { const g = (m, full) => ab.find((r) => r.component === c && r.model === m && (full ? r.variant === "full front-end" : r.variant !== "full front-end")); return [c, g("T2_physics", true).test, pm(g("T2_physics", true).acc, g("T2_physics", true).sd), pm(g("T2_physics", false).acc, g("T2_physics", false).sd), pm(g("HLR_clean", true).acc, g("HLR_clean", true).sd), pm(g("HLR_clean", false).acc, g("HLR_clean", false).sd)]; }),
  [2000, 1900, 1400, 1400, 1450, 1450], "Ablation (accuracy %, mean ± sd over 5 repeats).");
FIGURE("p3_ablation.png", "Ablation of the robust front-end components.", 0.85);
H2("7.5 What the noise study taught (development log)");
P("The robustness study was not only an evaluation; it found four flaws in earlier versions of the pipeline, each now fixed and documented:");
BUL([
  "**Detection threshold too high.** A midpoint off/on threshold made every trial \"off\" under moderate noise → replaced by the CFAR threshold (off class only).",
  "**Mains hum faked a fan comb** (50/100 Hz are harmonics of 25 Hz) → mains bands masked.",
  "**Harmonic-number ambiguity.** When noise left only the dominant line, the comb could not tell 3 × 34.5 Hz from 4 × 25.9 Hz and reported speed2 as speed1 → f0 search range restricted to the physically plausible 28–42 Hz.",
  "**Fragile feature selection** when a noise type was absent from training → invariant pool + worst-group objective (Section 6.4).",
]);

// =================================================================== 8 DEMO + MATLAB
H1("8. Demonstration and MATLAB port");
H2("8.1 Python command-line demo");
P("`py -m src.predict <recording.zip> [...]` classifies Phyphox exports with the saved final model and prints the class, the four probabilities, f0 (and rpm) and the number of fan harmonics; `--csv` writes a table and `--noise FAMILY LEVEL` first corrupts the recording with any calibrated noise model (for a live robustness demonstration). Running it on all 80 exports:");
D.demo_log.forEach((l) => body.push(new Paragraph({ spacing: { before: 0, after: 0, line: 220, lineRule: "exact" }, shading: { type: ShadingType.CLEAR, fill: "F4F4F4" }, children: [new TextRun({ text: l || " ", font: MONO, size: 15 })] })));
body.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
P(`Result: ${D.demo.correct}/${D.demo.n} correct, confidence in the predicted class ${pm(D.demo.conf_mean, D.demo.conf_sd)} % (minimum ${pct(D.demo.conf_min)}). **These are in-sample predictions** — the final model was trained on these recordings — so they demonstrate the tool, not the accuracy. The honest, out-of-sample figure is the LORO result: ${D.demo.loro_correct}/80 with true-class probability ${pm(D.demo.loro_ptrue_mean, D.demo.loro_ptrue_sd)} %.`);
TABLE(["Class", "Confidence (mean ± sd, %)", "f0 (Hz, mean ± sd)", "harmonics ≥ 6 dB (mean)"],
  D.demo.per_class.map((r) => [r.cls, pm(r.conf_mean, r.conf_sd), r.cls === "off" ? "— (noise)" : `${f(r.f0_mean)} ± ${f(r.f0_sd)}`, f(r.nh_mean, 2)]),
  [1800, 2700, 2600, 2500], "Demo predictions on the 80 recordings, per class.");
TABLE(["Calibrated noise added", "speed2_05.zip → prediction", "off_05.zip → prediction"],
  D.noise_demo.map((r) => [r.noise, `${r.speed2[0]} (${r.speed2[1].match(new RegExp(r.speed2[0] + "\\s+([\\d.]+%)"))[1]})`, `${r.off[0]} (${r.off[1].match(new RegExp(r.off[0] + "\\s+([\\d.]+%)"))[1]})`]),
  [3200, 3200, 3200], "`--noise` demonstrations (level units as in the noise-family table). The two failures are the documented physical limits: a compressor tone on speed1's line, and a 4 % speed shift (half the class spacing).");
H2("8.2 MATLAB port");
P("The final model was ported to MATLAB (folder `matlab/`, written for R2018a). It reads the same `final_model.json`, which contains every pipeline constant and both logistic-regression stages, and re-implements exactly the part of the pipeline the final model needs: clock regularisation, Hampel despiking, Kaiser-sinc resampling, zero-phase Butterworth high-pass, gap-aware median-Welch vector PSD, running-median floor, harmonic comb, and the two-stage logistic regression. Care was needed where MATLAB and NumPy differ: `max(NaN, 0)` returns 0 in MATLAB but NaN is kept by NumPy's `clip`, NumPy's 'reflect' padding excludes the edge sample, and `numpy.arange` excludes its end point.");
BUL([
  "`vib_predict(file)` — classify one recording; returns label, probabilities and features.",
  "`vib_demo` — classify every export in the project, print as the Python CLI does, write results/matlab_predictions.csv and verify against Python.",
  "`vib_preprocess`, `vib_features`, `vib_read`, `vib_load_model` — the building blocks.",
]);
TABLE(["Verification (80 recordings)", "Result"], [
  ["max |f0 difference|", `${D.matlab.df0} Hz`], ["max |comb-SNR difference|", `${D.matlab.dcs} dB`],
  ["harmonic-count mismatches", D.matlab.nh], ["max |probability difference|", D.matlab.dp],
  ["identical predictions", D.matlab.same], ["run time, all 80 recordings", `${f(D.matlab.seconds, 1)} s`], ["verdict", `**${D.matlab.result}**`],
], [5000, 4600], "MATLAB R2018a vs Python (run with `vib_demo`).");
P("The residual differences (10⁻⁸ Hz) come from the filter start-up conditions of MATLAB's and SciPy's `filtfilt`, which differ only inside the discarded 1 s margins. The full MATLAB code is listed in Appendix B.");

// =================================================================== 9 DISCUSSION
H1("9. Discussion");
H2("9.1 Why physics-informed features win");
P("The fan's identity is written in *frequencies*, which no mounting, gain or orientation change can alter; its *amplitude* is written by the mounting. Features built as ratios to the local noise floor (comb SNR, number of harmonics) and as frequencies (f0) inherit that invariance, and combining all harmonics into one comb turns several weak pieces of evidence into one strong one while letting no single foreign line dominate. Learning on noise-augmented data then mostly decides *where* to put two thresholds — and it puts them where physics says they belong (Section 6.5). This is why a two-feature logistic regression can be both 100 % accurate and robust, and why it stays interpretable.");
H2("9.2 Limitations and threats to validity");
BUL([
  "**One fan, one phone, one mount, one evening.** Cross-validation measures repeatability within that session; day-to-day supply-voltage changes are the main threat (Section 7.3). The calibrated other-day mix (≈ 88 %) is the best available estimate of cross-day performance.",
  "**Calibrated, not measured, disturbances.** Footstep and knock amplitudes rest on literature anchors; re-taping effects and the appliance amplitude are assumptions. All are flagged in Section 5.5.",
  "**Feature calibration used all 80 trials** to set realistic noise ranges (not to fit the classifier). This is a mild form of using test data for design; the leave-one-group-out protocol is the safeguard for the classifier itself.",
  "**Rotation hypothesis** rests on spectral evidence; a blade count and an independent rpm measurement are still needed.",
  "**The demo predictions are in-sample**; the reported accuracy is always the cross-validated one.",
]);
H2("9.3 Future work");
BUL([
  "Record a second session on another day after re-taping, with the mains voltage noted, and test the frozen model on it.",
  "Count the blades and measure rpm with a slow-motion video to close the rotation question.",
  "Record a few trials with a fridge/AC running and with footsteps, to calibrate the two uncalibrated disturbance amplitudes.",
  "A per-deployment calibration (one recording per speed) to absorb supply-voltage drift; a reject option (\"uncertain\") when P(on) or the speed probability is low.",
  "Live classification from a Phyphox remote stream (0.5–1 s windows are sufficient, Section 6.5).",
]);

// =================================================================== 10 CONCLUSION
H1("10. Conclusion");
P(`A phone's MEMS accelerometer is a sufficient instrument to recognise the operating state of a wall fan, provided the signal processing respects the instrument. Correct timing (hardware-clock fit), band-limited resampling (linear interpolation fakes a ${f(100 * (1 - p1.linear_amp_min), 0)} % amplitude decay), zero-phase gravity removal and median-averaged, orientation-invariant spectra expose a harmonic comb whose fundamental separates the three speeds with a margin of ≈ 7 % against a within-class spread of < 1 %. A two-stage logistic regression on just two of those physics features — how many fan harmonics are present, and at what fundamental — classifies all 80 recordings correctly out of sample (95 % CI ≥ 95.5 %; repeated CV ${pm(cv.HLR_aug.rep5x10_mean, cv.HLR_aug.rep5x10_sd)} %), remains at 100 % under every calibrated realistic household disturbance and degrades only where physics says it must. The MATLAB port reproduces it to 10⁻⁸ Hz.`);

// =================================================================== REFERENCES
H1("References");
[
  "T. B. Gabrielson, \"Mechanical-thermal noise in micromachined acoustic and vibration sensors,\" IEEE Trans. Electron Devices, 40(5), 903–909, 1993. https://pure.psu.edu/en/publications/mechanical-thermal-noise-in-micromachined-acoustic-and-vibration-/",
  "A. Stisen et al., \"Smart devices are different: assessing and mitigating mobile sensing heterogeneities for activity recognition,\" ACM SenSys 2015. https://portal.findresearcher.sdu.dk/da/publications/smart-devices-are-different-assessing-and-mitigating-mobile-sensi/",
  "TDK InvenSense, ICM-45605 datasheet (accelerometer noise 70 µg/√Hz). https://en.sekorm.com/doc/528502631.html",
  "TDK InvenSense, SmartMotion product flyer (ICM-42688-P noise density). https://invensense.tdk.com/wp-content/uploads/2024/12/tdk-invensense-flyers_smartmotion.pdf",
  "TDK InvenSense, ICM-42605 datasheet. https://yggdrasil.ti.bfh.ch/_static/datasheets/yggdrasil/ICM-42605.pdf",
  "TDK InvenSense, ICM-40609-D product page. https://invensense.tdk.com/ja-jp/products/6-axis/icm-40609-d",
  "PCBSync, MPU-6000 selection guide (noise 400 µg/√Hz). https://pcbsync.com/?p=34925",
  "AISC Design Guide 11, Vibrations of Steel-Framed Structural Systems Due to Human Activity (walking criterion 0.5 % g for offices/residences).",
  "Ding et al., \"Vibration test and comfort analysis of environmental and impact excitation for wooden floor structure,\" BioResources 15(4), 2020. https://bioresources.cnr.ncsu.edu/wp-content/uploads/2020/09/BioRes_15_4_8212_Ding_ZWGZH_Vibration_Test_Comfort_Analy_Environ_Excitat_Wood_Floor_Modif_17746.pdf",
  "G. R. Watts, Response of building elements to simulated traffic vibrations (door slams among domestic activities). https://www.ioa.org.uk/system/files/publications/GR%20WATTS%20RESPONSE%20OF%20BUILDING%20ELEMENTS%20TO%20SIMULATED%20TRAFFIC%20VIBRATIONS.pdf",
  "Prayas (Energy Group), Electricity Supply Monitoring Initiative — supply quality and voltage bands. https://www.prayaspune.org/peg/quality-of-electricity-supply ; https://energy.prayaspune.org/how-does-esmi-work",
  "Delhi Electricity Supply Code and Performance Standards Regulations, 2007 (±6 % LT voltage). https://www.latestlaws.com/mobile/app/bare-acts/70765",
  "Android sensor rate limiting (200 Hz without HIGH_SAMPLING_RATE_SENSORS): Expo sensors documentation. https://docs.expo.dev/versions/latest/sdk/sensors.md",
  "Phyphox — physical phone experiments, RWTH Aachen. https://phyphox.org",
].forEach((r, i) => body.push(new Paragraph({ spacing: { after: 80 }, children: runs(`[${i + 1}] ${r}`, { size: 19 }) })));

// =================================================================== APPENDICES
H1("Appendix A. Reproducibility");
TABLE(["Command", "What it does"], [
  ["py run_all.py", "Full pipeline data/raw → results/ (≈ 8 min on 16 cores): audit, features, calibration, classification, robustness, selection study, SUMMARY.md"],
  ["py run_all.py --quick", "Robustness study with 2 instead of 5 repeats"],
  ["py -m src.predict *.zip [--csv out.csv] [--noise FAMILY LEVEL]", "Demo classifier"],
  ["py report/build_report_data.py && node report/build_report.js", "Rebuild this report from the results (needs npm package docx)"],
  ["matlab: >> cd matlab; vib_demo", "MATLAB classification of all exports + verification against Python"],
], [4200, 5400], "Commands.", { left: true });
TABLE(["Component", "Version"], Object.entries(D.env).map(([k, v]) => [k, String(v)]), [3000, 6600], "Software environment.", { left: true });
P("All randomness (noise draws, augmentation, CV splits) is seeded from a hash of (task, trial, copy), so every run reproduces bit for bit. Data layout: `data/raw/<class>/<class>_NN.csv` (unpacked from the Phyphox zips, whose `meta/` folders supply start times and device information). Results: `results/feature_table.csv`, `results/tables/*.csv`, `results/figures/*.png`, `results/model/final_model.json`, `results/SUMMARY.md`.");

H1("Appendix B. MATLAB code (full)");
P("Written for MATLAB R2018a + Signal Processing Toolbox; verified against the Python implementation (Section 8.2).");
[["matlab/vib_predict.m", "vib_predict — classify one recording"], ["matlab/vib_preprocess.m", "vib_preprocess — Phase-1 chain"], ["matlab/vib_features.m", "vib_features — median-Welch PSD and harmonic comb"], ["matlab/vib_read.m", "vib_read — read a Phyphox export"], ["matlab/vib_load_model.m", "vib_load_model — load the JSON model"], ["matlab/vib_demo.m", "vib_demo — batch classification and verification"]].forEach(([fl, t]) => CODE(fl, t));

H1("Appendix C. Python code (full)");
[["run_all.py", "Entry point"], ["src/config.py", "Configuration and justification of every constant"], ["src/load_trials.py", "Loading and acquisition audit"], ["src/preprocess.py", "Phase 1 preprocessing"], ["src/features.py", "Phase 2 features"], ["src/calibrate_noise.py", "Noise-model calibration"], ["src/noise_models.py", "Noise models and augmentation"], ["src/ml.py", "Models, feature selection, CV, statistics"], ["src/workers.py", "Parallel feature computation"], ["src/phase1_audit.py", "Phase 1 driver and figures"], ["src/phase2_features.py", "Phase 2 driver and figures"], ["src/phase3_classify.py", "Phase 3a: classification and validation"], ["src/phase3_robustness.py", "Phase 3b: robustness and ablation"], ["src/phase3_selection_study.py", "Phase 3c: feature-selection design study"], ["src/predict.py", "Demo command-line classifier"], ["src/report.py", "results/SUMMARY.md generator"], ["report/build_report_data.py", "Data collection for this report"]].forEach(([fl, t]) => CODE(fl, t));

H1("Appendix D. Full robustness results");
P("Accuracy (%, mean ± sd over 5 repeated 5-fold CVs) for every tested level. R = level inside the calibrated realistic range.");
for (const fm of fams) {
  const rows = [];
  const lv = [...new Set(D.rob_full.filter((r) => r.family === fm).map((r) => r.level))];
  for (const L of lv) {
    const g = (m) => D.rob_full.find((r) => r.family === fm && r.level === L && r.model === m);
    const real = g("T2_physics").realistic ? "R" : "";
    rows.push([L === null ? "clean" : String(L), real, ...["T1_textbook", "T2_physics", "HLR_aug", "HLR_aug_known"].map((m) => pm(g(m).acc, g(m).sd, 0))]);
  }
  TABLE([`${fm} (${D.families[fm].unit})`, "R", "T1 textbook", "T2 physics", "HLR_aug (unseen)", "HLR_aug_known"], rows, [2700, 500, 1600, 1600, 1600, 1600], null, { size: 15 });
}

// ------------------------------------------------------------------ document
const doc = new Document({
  creator: "Project Computation Course", title: "Smartphone-Based Vibration Signature Analysis — Final Report",
  description: "Final report: fan state classification from a smartphone accelerometer",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 32, bold: true, color: "1F3864" }, paragraph: { spacing: { before: 240, after: 160 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 26, bold: true, color: "2F5597" }, paragraph: { spacing: { before: 240, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 22, bold: true, color: "404040" }, paragraph: { spacing: { before: 180, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [{ reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
    style: { paragraph: { indent: { left: 500, hanging: 260 } } } }] }] },
  sections: [{
    properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1134, bottom: 1134, left: 1134, right: 1134 } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: "Smartphone vibration signature analysis — final report", font: FONT, size: 16, color: "808080" })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: ["Page ", PageNumber.CURRENT, " of ", PageNumber.TOTAL_PAGES], font: FONT, size: 16, color: "808080" })] })] }) },
    children: body,
  }],
});
const out = path.join(ROOT, "report", "Final_Report.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log(`wrote ${out}: ${figN} figures, ${tabN} tables, ${body.length} blocks`); });
