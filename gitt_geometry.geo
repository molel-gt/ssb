//SetFactory('OpenCASCADE');
Mesh.CharacteristicLengthMin = 0.001;
Mesh.CharacteristicLengthMax = 0.1;

Point(1) = {0, 0, 0};
Point(2) = {10, 0, 0};
Point(3) = {10, 10, 0};
Point(4) = {0, 10, 0};

Point(5) = {5, 5, 0};
Point(6) = {5.5, 5, 0};
Point(7) = {5.5, 5.5, 0};
Point(8) = {4.5, 5, 0};
Point(9) = {4.5, 4.5, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Circle(5) = {6, 7, 8};
Circle(6) = {8, 9, 6};
//Circle(7) = {9, 5, 8};
//Circle(8) = {8, 5, 7};

//Curve Loop(1) = {1:4};
Line Loop(2) = {5:6};
Physical Line(1) = {4};
Physical Line(2) = {2};
Physical Line(3) = {1, 3};

Plane Surface(1) = {2};
//Plane Surface(1) = {1};
Physical Surface(1) = {1};