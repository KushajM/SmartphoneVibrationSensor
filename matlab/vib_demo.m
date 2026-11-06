%VIB_DEMO  Classify every Phyphox export in the project and verify the MATLAB port.
%   Run from the matlab/ folder:   >> vib_demo
%   1. classifies all <class>_NN.zip files (project root and data/zips), printing the class,
%      the four probabilities, f0 / rpm and the number of fan harmonics (same as src/predict.py)
%   2. writes results/matlab_predictions.csv
%   3. compares features and probabilities with results/model/reference_predictions.csv
%      (written by the Python pipeline) and prints PASS / FAIL with the largest differences
%   Requires R2016b+ and the Signal Processing Toolbox (butter, zp2sos, filtfilt, hann, sinc).

here = fileparts(mfilename('fullpath'));
root = fullfile(here, '..');
model = vib_load_model();
classes = model.classes;
fprintf('model: %s\n', model.model);
fprintf('       stage A features: %s | stage B features: %s\n\n', ...
        strjoin(cellstr(model.stage_a.features), ', '), strjoin(cellstr(model.stage_b.features), ', '));

files = [dir(fullfile(root, '*.zip')); dir(fullfile(root, 'data', 'zips', '*.zip'))];
n = numel(files);
names = cell(n, 1); truth = cell(n, 1); pred = cell(n, 1);
PR = zeros(n, 4); F0 = zeros(n, 1); NH = zeros(n, 1); CS = zeros(n, 1); LS = zeros(n, 1);
tic
for i = 1:n
    [~, stem] = fileparts(files(i).name);
    names{i} = stem;
    parts = strsplit(stem, '_');
    truth{i} = parts{1};
    [pred{i}, p, F] = vib_predict(fullfile(files(i).folder, files(i).name), model);
    PR(i, :) = p';  F0(i) = F.f0;  NH(i) = F.n_harm;  CS(i) = F.comb_snr;  LS(i) = F.line_snr;
    fprintf('%s\n  -> %-7s | off %5.1f%%  speed1 %5.1f%%  speed2 %5.1f%%  speed3 %5.1f%%\n', ...
            files(i).name, upper(pred{i}), 100 * p);
    if strcmp(pred{i}, 'off')
        fprintf('     no fan harmonic >= 6 dB above the noise floor (comb SNR %.1f dB)\n', F.comb_snr);
    else
        fprintf(['     f0 = %.2f Hz (%.0f rpm if f0 = shaft rate), dominant line %.1f Hz, ' ...
                 'harmonics >= 6 dB: %d, comb SNR %.1f dB\n'], F.f0, 60 * F.f0, 3 * F.f0, F.n_harm, F.comb_snr);
    end
end
fprintf('\n%d recordings in %.1f s\n', n, toc);
known = ismember(truth, classes);
conf = max(PR, [], 2);
fprintf('%d/%d correct (in-sample: the final model was trained on these trials);\n', ...
        sum(strcmp(truth(known), pred(known))), sum(known));
fprintf('confidence of predicted class: mean %.1f%% +- %.1f%%, min %.1f%%\n', ...
        100 * mean(conf), 100 * std(conf, 1), 100 * min(conf));

T = table(names, truth, pred, PR(:, 1), PR(:, 2), PR(:, 3), PR(:, 4), F0, NH, CS, LS, ...
          'VariableNames', {'trial', 'true_class', 'predicted', 'p_off', 'p_speed1', 'p_speed2', ...
                            'p_speed3', 'f0', 'n_harm', 'comb_snr', 'line_snr'});
writetable(T, fullfile(root, 'results', 'matlab_predictions.csv'));

% ------------------------------------------------------------------ verification vs Python
refFile = fullfile(root, 'results', 'model', 'reference_predictions.csv');
if exist(refFile, 'file')
    R = readtable(refFile);
    [found, loc] = ismember(R.trial, names);
    R = R(found, :);  loc = loc(found);
    dF0 = max(abs(R.f0 - F0(loc)));
    dCS = max(abs(R.comb_snr - CS(loc)));
    dNH = sum(R.n_harm ~= NH(loc));
    Pref = [R.p_off, R.p_speed1, R.p_speed2, R.p_speed3];
    dP = max(max(abs(Pref - PR(loc, :))));
    same = sum(strcmp(R.pred_final_in_sample, pred(loc)));
    fprintf('\nVerification against Python (%d trials):\n', height(R));
    fprintf('  max |f0 difference|        = %.2e Hz\n', dF0);
    fprintf('  max |comb SNR difference|  = %.2e dB\n', dCS);
    fprintf('  harmonic-count mismatches  = %d\n', dNH);
    fprintf('  max |probability diff|     = %.2e\n', dP);
    fprintf('  identical predictions      = %d/%d\n', same, height(R));
    if dF0 < 1e-2 && dCS < 0.05 && dNH == 0 && dP < 1e-2 && same == height(R)
        fprintf('  RESULT: PASS - the MATLAB port reproduces the Python model\n');
    else
        fprintf('  RESULT: FAIL - see the differences above\n');
    end
end
