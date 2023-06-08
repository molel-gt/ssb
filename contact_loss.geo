eps = 0.1;
Lx = 15.0;
Ly = 15.0;
Lz = 15.0;
SetFactory('OpenCASCADE');
R = Lx * Sqrt(eps/Pi);
xc = 0.5 * Lx;
yc = 0.5 * Ly;
xc1 = 0.3 * Lx;
xc2 = 0.7 * Lx;
//Mesh.CharacteristicLengthMin = 0.05;
//Mesh.CharacteristicLengthMax = 0.05;
//General.NumThreads = 16;
//Mesh.MaxNumThreads1D = 16;
//Mesh.MaxNumThreads2D = 16;
//Mesh.MaxNumThreads3D = 16;
Point(1) = {0, 0, 0};
Point(2) = {Lx, 0, 0};
Point(3) = {Lx, Ly, 0};
Point(4) = {0, Ly, 0};
Point(5) = {0, 0, Lz};
Point(6) = {Lx, 0, Lz};
Point(7) = {Lx, Ly, Lz};
Point(8) = {0, Ly, Lz};

Point(9) = {xc1, yc, 0};
Point(10) = {xc1 + R, yc, 0};
Point(11) = {xc1, yc + R, 0};
Point(12) = {xc1 - R, yc, 0};
Point(13) = {xc1, yc - R, 0};

Point(14) = {xc2, yc, 0};
Point(15) = {xc2 + R, yc, 0};
Point(16) = {xc2, yc + R, 0};
Point(17) = {xc2 - R, yc, 0};
Point(18) = {xc2, yc - R, 0};

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

Circle(13) = {10, 9, 11};
Circle(14) = {11, 9, 12};
Circle(15) = {12, 9, 13};
Circle(16) = {13, 9, 10};

Circle(17) = {15, 14, 16};
Circle(18) = {16, 14, 17};
Circle(19) = {17, 14, 18};
Circle(20) = {18, 14, 15};

Curve Loop(1) = {13, 14, 15, 16};
Plane Surface(1) = {1};
Curve Loop(8) = {17, 18, 19, 20};
Plane Surface(8) = {8};
Curve Loop(7) = {1, 2, 3, 4};
Plane Surface(7) = {7, 1, 8};
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

Physical Surface(1) = {1, 8};
Physical Surface(2) = {2};
Physical Surface(3) = {3, 4, 5, 6, 7};
Surface Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};
Volume(1) = {1};
Physical Volume(1) = {1};
// refining
//Characteristic Length {1:6, 9, 10, 11, 12, 13} = 0.05;
Coherence Mesh;
