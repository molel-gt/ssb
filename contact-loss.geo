cov = 0.05;
L = 50;
SetFactory('OpenCASCADE');
R = 2 * L * (cov/Pi) ^ (0.5);
xc = 0.5 * L;
yc = 0.5 * L;
Mesh.CharacteristicLengthMin = 0.1;
Mesh.CharacteristicLengthMax = 0.5;
Point(1) = {0, 0, 0};
Point(2) = {L, 0, 0};
Point(3) = {L, L, 0};
Point(4) = {0, L, 0};
Point(5) = {0, 0, L};
Point(6) = {L, 0, L};
Point(7) = {L, L, L};
Point(8) = {0, L, L};

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

Physical Surface(1) = {1};
Physical Surface(2) = {2};
Physical Surface(3) = {3, 4, 5, 6, 7};
Surface Loop(1) = {1, 2, 3, 4, 5, 6, 7};
Volume(1) = {1};
Physical Volume(1) = {1};

// Transfinite Line "*" = 100 Using Bump 0.25;
// Transfinite Surface "*";
// Recombine Surface "*";
// Transfinite Volume "*";

Field[1] = Distance;
Field[2] = Threshold;
//Field[1].PointsList = {109};
Field[1].FacesList = {1, 2, 3, 4, 5, 6, 7};
Field[2].IField = 1;
Field[2].LcMin = 0.001;
Field[2].LcMax = 0.01;
Field[2].DistMin = 0;
Field[2].DistMax = 5;
// Field[1].Sampling = 100;
Mesh.Algorithm = 5;
Mesh.Smoothing = 5;
