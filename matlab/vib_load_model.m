function model = vib_load_model(jsonFile)
%VIB_LOAD_MODEL  Load the final classifier exported by the Python pipeline (run_all.py).
%   model = vib_load_model()            reads ../results/model/final_model.json
%   model = vib_load_model(jsonFile)    reads the given file
%
%   The JSON holds every pipeline constant (sampling rate, filter, comb parameters) and both
%   logistic-regression stages (feature names, standardisation, coefficients), so MATLAB and
%   Python apply exactly the same model.  Requires R2016b+ (jsondecode).
if nargin < 1 || isempty(jsonFile)
    here = fileparts(mfilename('fullpath'));
    jsonFile = fullfile(here, '..', 'results', 'model', 'final_model.json');
end
model = jsondecode(fileread(jsonFile));
end
