function area=SurfArea(V,A1)
%function area=SurfArea(V,A1=0.6380)
%
% Surface area estimation for 3D binary objects
% Unbiased with minimum variance for planar surfaces of random orientation
%
% Input: Binary 3D volume V and
%  optional Surface area of m-cube case 1 (default A1=0.6380)
%
%  All non-zero voxels are concidered to be object
%  Surface toward voxels outside the volume is not included
%
% Copyright (C) 2004-2010 Joakim Lindblad
%  Not for distribution! Newer, public files will
%  be available at http://www.cb.uu.se/~joakim/software/
%
%
% References
% ----------
% J. Lindblad.
% Surface Area Estimation of Digitized 3D Objects using Weighted Local Configurations.
% Image and Vision Computing, Vol. 23, No. 2, pp. 111-122, 2005.
% doi:10.1016/j.imavis.2004.06.012
%
% J. Lindblad.
% Surface Area Estimation of Digitized Planes Using Weighted Local Configurations.
% In Proc. of the 11th International Conference on Discrete Geometry for Computer Imagery (DGCI)
% LNCS-2886, pp. 348-357, Naples, Italy, Nov. 2003.
% doi:10.1007/b94107


if exist('m_cube_histogram')~=3, error('Missing mex-file. Please run: mex m_cube_histogram.c'); end

% A1=0.2118 is the suggestion of the DGCI-2003 paper
% A1=0.6380 is the suggestion of the IVC-2004 paper
if nargin<2, A1=0.6380; end

% Binary only
V=logical(V);

% Unbiased combination for planes, for any A1
A2=0.6690;
A5=1.1897-A1;
A8=0.9270;
A9=1.6942-2*A1;

% A11 is non-planar
A11=1.573132; %from Marching Cubes triangulation

% Table of elementary surface areas
areatab=[0; A1; A2; 2*A1; 2*A1; A5; A1+A2; 3*A1; A8; A9; 2*A2; A11; A1+A5; 4*A1];

% m-cube lookup table, 128 cases (remaining part is mirror symmetric)
m_case = uint8([ ...
0,1,1,2,   1,2,3,5,   1,3,2,5,   2,5,5,8,   ... %0-15
1,2,3,5,   3,5,7,9,   4,6,6,11,  6,11,12,5, ... %16
1,3,2,5,   4,6,6,11,  3,7,5,9,   6,12,11,5, ... %32
2,5,5,8,   6,11,12,5, 6,12,11,5, 10,6,6,2,  ... %48
1,3,4,6,   2,5,6,11,  3,7,6,12,  5,9,11,5,  ... %64
2,5,6,11,  5,8,12,5,  6,12,10,6, 11,5,6,2,  ... %80
3,7,6,12,  6,12,10,6, 7,13,12,7, 12,7,6,3,  ... %96
5,9,11,5,  11,5,6,2,  12,7,6,3,  6,3,4,1 ]);    %112-127
m_case = [m_case,fliplr(m_case)]; %Symmetric

% 256 cases area table
areatab2=areatab([m_case+1]);

% m-cube histogram
h=m_cube_histogram(V);

% Surface area
area=sum(h.*areatab2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Function to calculate a histogram of m-cubes
%% Input : binary image volume
%% Output: number of occurences of each m-cube type in the volume
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Slow matlab-version, in case mexing is not an option
%{
function h=m_cube_histogram(V) 

code=zeros(size(V)-1,'uint8'); % m-code volume
code=bitset(code,1,V(1:end-1,1:end-1,1:end-1)); %1
code=bitset(code,2,V(1:end-1,2:end  ,1:end-1)); %2 x
code=bitset(code,3,V(2:end  ,1:end-1,1:end-1)); %3 y
code=bitset(code,4,V(2:end  ,2:end  ,1:end-1)); %4 xy
code=bitset(code,5,V(1:end-1,1:end-1,2:end  )); %5 z
code=bitset(code,6,V(1:end-1,2:end  ,2:end  )); %6 xz
code=bitset(code,7,V(2:end  ,1:end-1,2:end  )); %7 yz
code=bitset(code,8,V(2:end  ,2:end  ,2:end  )); %8 xyz

h=histc(code(:),0:255);
%}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                               END OF FILE                               %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
