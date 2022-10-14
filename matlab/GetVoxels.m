function [M] = GetVoxels(phase)
    allFiles = {dir(strcat('Archive/', phase, '/*.bmp')).name};
    data047 = imread(strcat('Archive/', phase, '/SegIm47.bmp'));
    data047 = data047(:, :, 1);
    [ny, nz] = size(data047);
    nx = length(allFiles);
    disp([nx, ny, nz])
    M = zeros(nx, ny, nz);
    for i=1:nx
        odata = imread(strcat("Archive/", phase, '/', allFiles{i}));
        data = odata(:, :, 1);
        data = data / 255;
        M(i, :, :) = data;
    end
end