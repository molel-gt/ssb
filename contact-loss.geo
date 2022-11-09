R = 15;
xc = 25;
yc = 25;

Point(1) = {0, 0, 0};
Point(2) = {50, 0, 0};
Point(3) = {50, 50, 0};
Point(4) = {0, 50, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};
Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};
Extrude {0, 0, 50} { Surface{1};}
Physical Volume(1) = {1};

Point(109) = {xc, yc, 0};
Point(110) = {xc + R, yc, 0};
Point(111) = {xc, yc + R, 0};
Point(112) = {xc - R, yc, 0};
Point(113) = {xc, yc - R, 0};

Point(115) = {0, 0, 50};
Point(116) = {50, 0, 50};
Point(117) = {50, 50, 50};
Point(118) = {0, 50, 50};

Line(100) = {115, 116};
Line(101) = {116, 117};
Line(102) = {117, 118};
Line(103) = {118, 115};
Curve Loop(2) = {100, 101, 102, 103};
Plane Surface(2) = {2};
Physical Surface(2) = {2};

Circle(22) = {110, 109, 111};
Circle(23) = {111, 109, 112};
Circle(24) = {112, 109, 113};
Circle(25) = {113, 109, 110};

Curve Loop(3) = {22, 23, 24, 25};
Plane Surface(3) = {3};
Physical Surface(1) = {3};