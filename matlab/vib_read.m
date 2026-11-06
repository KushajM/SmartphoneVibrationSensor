function [t, a] = vib_read(file)
%VIB_READ  Read a Phyphox "Acceleration (with g)" recording.
%   [t, a] = vib_read('speed2_05.zip')   Phyphox CSV export (zip containing "Raw Data.csv")
%   [t, a] = vib_read('speed2_05.csv')   the extracted "Raw Data.csv"
%   t : N x 1 timestamps [s];  a : N x 3 acceleration x, y, z including gravity [m/s^2]
[~, ~, ext] = fileparts(file);
if strcmpi(ext, '.zip')
    tmp = tempname;
    mkdir(tmp);
    cleaner = onCleanup(@() rmdir(tmp, 's'));
    unzip(file, tmp);
    d = dlmread(fullfile(tmp, 'Raw Data.csv'), ',', 1, 0);     % skip the quoted header row
else
    d = dlmread(file, ',', 1, 0);
end
t = d(:, 1);
a = d(:, 2:4);
end
