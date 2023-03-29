SetFactory('OpenCASCADE');
//Mesh.CharacteristicLengthMin = 0.001;
//Mesh.CharacteristicLengthMax = 0.1;
// Mesh.SubdivisionAlgorithm  = 1;
Mesh.Hexahedra = 1;
Point(1) = {0, 0, 0};
Point(2) = {1e-2, 0, 0};  // 1 centimeters
Point(3) = {1e-2, 1e-5, 0};
Point(4) = {0, 1e-5, 0}; // 10 micrometers

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Physical Line(1) = {4};
Physical Line(2) = {2};
Physical Line(3) = {1, 3};
Curve Loop(1) = {1:4};
Plane Surface(1) = {1};
Physical Surface(1) = {1};