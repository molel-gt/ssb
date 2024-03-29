SetFactory('OpenCASCADE');
Mesh.CharacteristicLengthMin = 0.001;
Mesh.CharacteristicLengthMax = 0.1;
// Mesh.SubdivisionAlgorithm  = 1;
Mesh.Hexahedra = 1;
Point(1) = {0, 0, 0};
Point(2) = {20, 0, 0};
Point(3) = {20, 20, 0};
Point(4) = {0, 20, 0};
Point(5) = {0, 17.5, 0};
Point(6) = {19.5, 17.5, 0};
Point(7) = {19.5, 12.5, 0};
Point(8) = {0, 12.5, 0};
Point(9) = {0, 7.5, 0};
Point(10) = {19.5, 7.5, 0};
Point(11) = {19.5, 2.5, 0};
Point(12) = {0, 2.5, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 1};

Physical Line(1) = {12, 4:11};
Physical Line(2) = {2};
Physical Line(3) = {1, 3};
Curve Loop(1) = {1:12};
Plane Surface(1) = {1};
Physical Surface(1) = {1};