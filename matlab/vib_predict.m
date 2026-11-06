function [label, probs, F] = vib_predict(recording, model)
%VIB_PREDICT  Classify a Phyphox recording: off / speed1 / speed2 / speed3.
%   [label, probs, F] = vib_predict('speed2_05.zip')
%   [label, probs, F] = vib_predict(file, model)      model from vib_load_model (faster in loops)
%
%   label : 'off' | 'speed1' | 'speed2' | 'speed3'
%   probs : 4 x 1 [P(off); P(speed1); P(speed2); P(speed3)]
%   F     : comb features (f0, n_harm, comb_snr, line_snr)
%
%   Final model = two-stage logistic regression, trained in Python on the 80 real trials plus
%   calibrated noise-augmented copies (see the report):
%     stage A  P(fan on)       = sigmoid(w_A * standardise(features_A) + b_A)
%     stage B  P(speed k | on) = softmax(W_B * standardise(features_B) + b_B)
%     P(off) = 1 - P(on),  P(speed k) = P(on) * P(speed k | on);  off if P(on) < 0.5
if nargin < 2 || isempty(model)
    model = vib_load_model();
end
P = model.pipeline;
[t, a] = vib_read(recording);
dur = t(end) - t(1);
trim = P.trim_s;
if dur < P.trim_s + 1
    trim = max(1, dur - 1);                           % short recording: shorter window
end
[x, valid, fs_in] = vib_preprocess(t, a, P, trim);
F = vib_features(x, valid, fs_in, P);
pa = stage_prob(model.stage_a, F);
pa = pa(2);                                           % P(fan on)
pb = stage_prob(model.stage_b, F);                    % P(speed | on)
probs = [1 - pa; pa * pb(:)];
if pa < 0.5
    label = 'off';
else
    [~, k] = max(pb);
    label = model.classes{model.stage_b.classes(k) + 1};
end
end


function p = stage_prob(st, F)
names = st.features;
if ischar(names)
    names = {names};
end
z = zeros(numel(names), 1);
for i = 1:numel(names)
    if ~isfield(F, names{i})
        error('vib_predict:feature', 'feature "%s" is not computed by the MATLAB port', names{i});
    end
    z(i) = F.(names{i});
end
z = (z - st.scaler_mean(:)) ./ st.scaler_scale(:);
W = st.coef;
b = st.intercept(:);
if numel(b) == 1                                      % binary logistic regression
    if size(W, 1) ~= 1, W = W'; end
    p1 = 1 / (1 + exp(-(W * z + b)));
    p = [1 - p1; p1];
else                                                  % multinomial: softmax
    if size(W, 1) ~= numel(b), W = W'; end
    s = W * z + b;
    e = exp(s - max(s));
    p = e / sum(e);
end
end
