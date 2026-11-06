function F = vib_features(x, valid, fs_in, P)
%VIB_FEATURES  Harmonic-comb features used by the final model (port of src/features.py).
%   F = vib_features(x, valid, fs_in, P)
%   F.comb_snr : mean clipped SNR of the best harmonic comb k*f0, k = 1..4  [dB]
%   F.f0       : fundamental of that comb, refined to sub-bin accuracy     [Hz]
%   F.n_harm   : number of its harmonics >= 6 dB above the local floor
%   F.line_snr : SNR of its strongest harmonic                              [dB]
%
%   Spectrum: Welch, Hann 512 (0.78 Hz bins), 75 % overlap, MEDIAN across segments (rejects
%   transients such as footsteps and knocks), per-axis PSDs summed (= PSD of the acceleration
%   vector -> independent of phone orientation), segments touching dropped samples skipped.
%   Noise floor: running median over +-10 Hz;  SNR = 10 log10(P / floor).
%   Comb: f0 searched over 28-42 Hz (physical prior; excludes the k=2/k=4 aliases of the
%   dominant 3*f0 line), harmonics only below 0.45 x the device rate, +-1 Hz bands around
%   50/100/150 Hz (fixed-frequency mains hum) never count, each harmonic capped at 30 dB.
[f, Pw, ~, snr] = robust_psd(x, valid, P);
F = comb(f, Pw, snr, fs_in, P);
end


function [f, Pw, flo, snr] = robust_psd(x, valid, P)
n = size(x, 1);
nper = min(P.rob_nperseg, 2^floor(log2(max(floor(n / 2), 2))));
hop = max(floor(nper / 4), 1);
starts = 0:hop:(n - nper);
if ~all(valid)
    ok = arrayfun(@(s) mean(valid(s+1:s+nper)) >= 0.95, starts);
    if sum(ok) >= 3
        starts = starts(ok);
    end
end
w = hann(nper, 'periodic');
nf = nper / 2 + 1;
Pseg = zeros(numel(starts), nf);
for i = 1:numel(starts)
    seg = x(starts(i)+1:starts(i)+nper, :);
    seg = seg - mean(seg, 1);
    X = fft(seg .* w);
    Pseg(i, :) = (sum(abs(X(1:nf, :)).^2, 2) * (2 / (P.fs * sum(w.^2))))';
end
if numel(starts) >= 3
    Pw = median(Pseg, 1)';
else
    Pw = mean(Pseg, 1)';
end
f = (0:nf-1)' * (P.fs / nper);
L = 2 * round(P.floor_halfwidth_hz / (f(2) - f(1))) + 1;   % running-median length (odd)
h = (L - 1) / 2;
Pp = [repmat(Pw(1), h, 1); Pw; repmat(Pw(end), h, 1)];      % 'nearest' edge padding
flo = median(Pp((1:numel(Pw)) + (0:L-1)'), 1)';
snr = 10 * log10(max(Pw, 1e-30) ./ max(flo, 1e-30));
end


function F = comb(f, Pw, snr, fs_in, P)
s = snr;
for m = 1:3
    s(abs(f - m * P.mains_hz) <= P.mains_mask_hw) = NaN;
end
fmax = min(0.45 * fs_in, 0.95 * f(end));
grid = P.f0_min + (0:P.f0_n-1)' * P.f0_step;
ks = 1:P.k_max;
H = grid * ks;
inb = H < fmax;
Sraw = interp1(f, s, H, 'linear');
S = min(max(Sraw, 0), P.snr_cap_db);
S(isnan(Sraw) | ~inb) = NaN;                 % MATLAB max() would turn NaN into 0
n_in = sum(inb, 2);
n_mask = sum(inb & isnan(S), 2);
score = mean(S, 2, 'omitnan');
ok = (n_in - n_mask >= 1) & (n_mask <= 1) & isfinite(score);
if ~any(ok)
    F = struct('comb_snr', 0, 'f0', mean([P.f0_min P.f0_max]), 'n_harm', 0, 'line_snr', 0);
    return
end
score(~ok) = -Inf;
[~, i] = max(score);
f0c = grid(i);
est = [];
wts = [];
for k = ks
    if k * f0c >= fmax
        continue
    end
    j = find(abs(f - k * f0c) <= 1.2 & isfinite(s));
    if isempty(j)
        continue
    end
    [~, jm] = max(Pw(j));
    jj = j(jm);
    est(end+1) = parabolic(f, Pw, jj) / k;          %#ok<AGROW>
    wts(end+1) = max(s(jj), 0) + 1e-6;              %#ok<AGROW>
end
if isempty(est)
    f0 = f0c;
else
    f0 = sum(est .* wts) / sum(wts);
end
Si = S(i, :);
Si = Si(isfinite(Si));
F = struct('comb_snr', score(i), 'f0', f0, 'n_harm', sum(Si >= P.strong_db), ...
           'line_snr', 0);
if ~isempty(Si)
    F.line_snr = max(Si);
end
end


function fp = parabolic(f, Pw, kk)
% sub-bin peak location from a parabola through log-power at bins kk-1, kk, kk+1
if kk <= 1 || kk >= numel(Pw)
    fp = f(kk);
    return
end
y = log(max(Pw(kk-1:kk+1), 1e-30));
den = y(1) - 2 * y(2) + y(3);
if den ~= 0
    d = 0.5 * (y(1) - y(3)) / den;
else
    d = 0;
end
fp = f(kk) + min(max(d, -0.5), 0.5) * (f(2) - f(1));
end
