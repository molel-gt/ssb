SetFactory('OpenCASCADE');
R = 15;
xc = 25;
yc = 25;

Point(1) = {0, 0, 0};
Point(2) = {50, 0, 0};
Point(3) = {50, 50, 0};
Point(4) = {0, 50, 0};

Point(109) = {xc, yc, 0};
Point(110) = {xc + R, yc, 0};
Point(111) = {xc, yc + R, 0};
Point(112) = {xc - R, yc, 0};
Point(113) = {xc, yc - R, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};
Circle(22) = {110, 109, 111};
Circle(23) = {111, 109, 112};
Circle(24) = {112, 109, 113};
Circle(25) = {113, 109, 110};
Curve Loop(2) = {22, 23, 24, 25};
Plane Surface(1) = {1};
Physical Surface(1) = {1};
// Plane Surface(2) = {2};
Physical Surface(1) = {2};
// Plane Surface(3) = {1, 2};
// BooleanFragments{Surface{2}; }{Surface{3}; }
Extrude {0, 0, 50} { Surface{1}; }
// Physical Surface(1) = {2};
// getCenterOfMass Surface{1}
Physical Volume(1) = {1};