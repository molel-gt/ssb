nx = 200;
ny = 200;
nz = 202;
data = ones(nx+50, ny, nz);

gridX = 1:nx;
gridY = 1:ny;
gridZ = 1:nz;

for idx = 1:nz
  %img_file = ["~/work/ssb/output/segmentation/cam/" num2str(idx, "%03.0f") ".tif"];
  img_file = ["~/work/ssb/output/segmented/" num2str(idx) ".tif"];
  img = imread(img_file);
  img = img(1:nx, 1:ny, 1);
  data(1:nx, :, idx) = img;
end
% cam attachment to current collector
data(1:10, :, :) = 3;
% separator
data(nx:nx+50, :, :) = 2;
data = uint8(data);
dofix = 0;
method = "cgalmesh";

clear opt;
opt.autoregion = 1;
%isovalues = [1];
%maxvol = 2;
%opt(1).keepratio = 0.05; % resample levelset 1 to 5%
%opt(2).keepratio = 0.1;  % resample levelset 2 to 10%
opt(1).radbound = 1; % head surface element size bound
opt(2).radbound = 1; % brain surface element size bound
%opt(1).side = 'lower'; %
%opt(2).side = 'lower'; %

%[node, elem, face, regions] =  vol2mesh(data, gridX, gridY, gridZ, opt, [0, 10, 20], dofix, method);
[node, elem, face] = v2m(data, [1, 2, 3], opt, [], 'cgalmesh');
%clear opt;
%opt.radbound = 2;
%[node, elem, face] = v2m(uint8(data), 0.5, 1, [], 'cgalmesh');
%plotmesh(node, face);
%axis equal;
