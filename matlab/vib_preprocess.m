function [x, valid, fs_in] = vib_preprocess(t, a, P, trim)
%VIB_PREPROCESS  Phase-1 preprocessing (port of src/preprocess.py).
%   [x, valid, fs_in] = vib_preprocess(t, a, P, trim)
%   t, a  : raw timestamps and N x 3 acceleration (gravity included)
%   P     : model.pipeline (constants from final_model.json)
%   trim  : analysed window length [s] (5 s normally)
%   x     : gravity-free acceleration, exactly P.fs Hz, central `trim` seconds
%   valid : false where a sample lies in/near a dropped-sample gap
%   fs_in : the device's true sampling rate from the hardware-clock fit
%
%   Steps (order matters):
%   1. assign samples to integer slots of the sensor clock, regress time on slot index
%      -> true rate; dropped slots are bridged and flagged
%   2. Hampel despiker BEFORE resampling (a sinc kernel would smear one bad sample)
%   3. band-limited (Kaiser-windowed sinc) resampling to exactly P.fs.  Linear interpolation
%      is NOT used: at f/fs = 0.27 its gain at a half-sample offset is cos(pi f/fs) = 0.65 and
%      because 400.108 Hz and 400 Hz grids slide past each other it fakes an amplitude decay.
%   4. 4th-order Butterworth high-pass 2 Hz, forward-backward (zero phase) on the full record
%   5. keep the central `trim` seconds (filter start-up transients fall in the margins)

% ---------------------------------------------------------------- 1. clock slots
dt_med = median(diff(t));
slots = round((t - t(1)) / dt_med);
keep = [true; diff(slots) > 0];                     % drop duplicated slots
slots = slots(keep);  tt = t(keep);  aa = a(keep, :);
p = polyfit(slots, tt, 1);
fs_in = 1 / p(1);
n = slots(end) + 1;
missing = true(n, 1);
missing(slots + 1) = false;
xs = interp1(slots, aa, (0:n-1)', 'linear');        % identity on present slots, bridges gaps

% ---------------------------------------------------------------- 2. Hampel despiker
xs = hampel_local(xs, P.hampel_half, P.hampel_nsig);

% ---------------------------------------------------------------- 3. sinc resampling
dur = (n - 1) / fs_in;
step = 1 / P.fs;
n_out = ceil(dur / step);                            % = len(numpy.arange(0, dur, step))
t_out = (0:n_out-1)' * step;
y = sinc_eval(xs, t_out * fs_in, min(1, P.fs / fs_in), P.sinc_half, P.sinc_beta);
bad = conv(double(missing), ones(5, 1), 'same') > 0;
idx = min(max(round(t_out * fs_in), 0), numel(bad) - 1) + 1;
valid = ~bad(idx);

% ---------------------------------------------------------------- 4. gravity removal
[z, pz, k] = butter(P.hp_order, P.hp_fc / (P.fs / 2), 'high');
[sos, g] = zp2sos(z, pz, k);
xf = filtfilt(sos, g, y);

% ---------------------------------------------------------------- 5. central window
N = round(trim * P.fs);
i0 = max(floor((size(xf, 1) - N) / 2), 0);
x = xf(i0+1:i0+N, :);
valid = valid(i0+1:i0+N);
end


function y = sinc_eval(xs, pos, ratio, half, beta)
% Evaluate uniformly sampled xs at fractional sample positions pos (0-based) with a
% Kaiser-windowed sinc; ratio < 1 lowers the cut-off to ratio * Nyquist (anti-aliasing).
n_in = size(xs, 1);
hw = ceil(half / ratio);
i0 = floor(pos);
k = -hw+1:hw;
idx = i0 + k;                                        % n_out x 2hw  (implicit expansion)
d = pos - idx;
w = ratio * sinc(ratio * d) .* besseli(0, beta * sqrt(max(1 - (d / hw).^2, 0))) / besseli(0, beta);
w = w ./ sum(w, 2);                                  % exact unity DC gain
idx = min(max(idx, 0), n_in - 1) + 1;                % repeat edge samples
mu = mean(xs, 1);
y = zeros(numel(pos), size(xs, 2));
for c = 1:size(xs, 2)
    xc = xs(:, c) - mu(c);
    y(:, c) = sum(w .* xc(idx), 2) + mu(c);
end
end


function out = hampel_local(x, half, nsig)
% Hampel identifier: replace samples further than nsig robust sigmas from the local median.
% Reflect padding as numpy 'reflect'; a record-wide robust sigma is the floor so coarsely
% quantised signals (local MAD = 0) are left alone.
n = size(x, 1);
w = 2 * half + 1;
xp = [x(half+1:-1:2, :); x; x(end-1:-1:end-half, :)];
idx = (1:n) + (0:w-1)';                              % w x n window indices
out = x;
for c = 1:size(x, 2)
    xc = xp(:, c);
    W = xc(idx);
    med = median(W, 1)';
    mad = median(abs(W - med'), 1)';
    resid = x(:, c) - med;
    fl = 1.4826 * median(abs(resid - median(resid)));
    sigma = max(1.4826 * mad, fl);
    f = abs(resid) > nsig * max(sigma, 1e-12);
    out(f, c) = med(f);
end
end
