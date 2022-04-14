function [M] = GetVoxels()
    allFiles = {dir('Archive/activematerial/*.bmp').name};
    data047 = imread("Archive/activematerial/SegIm47.bmp");
    data047 = data047(:, :, 1);
    [ny, nz] = size(data047);
    nx = length(allFiles);
    disp([nx, ny, nz])
    M = zeros(nx, ny, nz);
    for i=1:nx
        odata = imread(strcat("Archive/activematerial/", allFiles{i}));
        data = odata(:, :, 1);
        data = data / 255;
        M(i, :, :) = data;
    end
end