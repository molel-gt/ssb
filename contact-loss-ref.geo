eps = 0.25;
Lx = 25.0;
Ly = 25.0;
Lz = 25.0;
lmax = 0.75;
mleft = 1;
mright = 2;
minsulated = 3;
SetFactory('OpenCASCADE');
R = Lx * Sqrt(eps/Pi);
xc = 0.5 * Lx;
yc = 0.5 * Ly;
General.ExpertMode = 1;
Mesh.CharacteristicLengthMax = lmax;
Point(1) = {0, 0, 0};
Point(2) = {Lx, 0, 0};
Point(3) = {Lx, Ly, 0};
Point(4) = {0, Ly, 0};
Point(5) = {0, 0, Lz};
Point(6) = {Lx, 0, Lz};
Point(7) = {Lx, Ly, Lz};
Point(8) = {0, Ly, Lz};

Point(109) = {xc, yc, 0};
Point(110) = {xc + R, yc, 0};
Point(111) = {xc, yc + R, 0};
Point(112) = {xc - R, yc, 0};
Point(113) = {xc, yc - R, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 5};

Line(9) = {5, 1};
Line(10) = {4, 8};
Line(11) = {2, 6};
Line(12) = {3, 7};

Circle(22) = {110, 109, 111};
Circle(23) = {111, 109, 112};
Circle(24) = {112, 109, 113};
Circle(25) = {113, 109, 110};
Curve Loop(1) = {22, 23, 24, 25};
Plane Surface(1) = {1};
Curve Loop(7) = {1, 2, 3, 4};
Plane Surface(7) = {7, 1};
Curve Loop(2) = {5, 6, 7, 8};
Plane Surface(2) = {2};
Curve Loop(3) = {4, 9, 8, 10};
Plane Surface(3) = {3};
Curve Loop(4) = {2, 11, 6, 12};
Plane Surface(4) = {4};
Curve Loop(5) = {3, 12, 7, 10};
Plane Surface(5) = {5};
Curve Loop(6) = {1, 11, 5, 9};
Plane Surface(6) = {6};

Physical Surface(mleft) = {1};
Physical Surface(mright) = {2};
Physical Surface(minsulated) = {3, 4, 5, 6, 7};
Surface Loop(1) = {1, 2, 3, 4, 5, 6, 7};
Volume(1) = {1};
Physical Volume(1) = {1};
// refining
Characteristic Length {1:6, 109, 110, 111, 112, 113} = lmax * (1 + 3 * Lz/100);
